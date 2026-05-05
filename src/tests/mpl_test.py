"""Tests for src.lusca.mpl_freeze."""

import logging
import os
import subprocess
import textwrap
from datetime import datetime

import numpy as np
import pytest

from src.lusca.mpl_freeze import (
    _check_free_names,
    _parse_line,
    _save_npz,
    _smoke_test_replot,
    _write_metadata,
    _write_replot,
)


def test_parse_line():
    """Test with a simple command."""
    line = "demo_plot var1 var2 --outdir custom_dir"
    name, vars, outdir = _parse_line(line)

    assert name == "demo_plot"
    assert vars == ["var1", "var2"]
    assert outdir == "custom_dir"

    # Test with default output directory
    line = "demo_plot var1 var2"
    name, vars, outdir = _parse_line(line)

    assert name == "demo_plot"
    assert vars == ["var1", "var2"]
    assert outdir == "docs/figs"

    # Test with no variables
    line = "demo_plot"
    name, vars, outdir = _parse_line(line)

    assert name == "demo_plot"
    assert vars == []
    assert outdir == "docs/figs"


# ---- Bug 1: pickle save/load asymmetry ----


def test_save_npz_rectangular_roundtrip(tmp_path, caplog):
    """Bug 1a: rectangular numeric arrays save without warning and load cleanly."""
    ns = {"x": np.linspace(0, 1, 5), "y": np.arange(10).reshape(2, 5)}
    npz = tmp_path / "rect.npz"

    with caplog.at_level(logging.WARNING, logger="root"):
        info = _save_npz(npz, ns, ["x", "y"])

    assert "dtype=object" not in caplog.text
    assert info["x"]["dtype"] != object
    assert info["y"]["dtype"] != object

    # Loads without allow_pickle (the strict default) — no pickle was used.
    data = np.load(npz)
    assert np.allclose(data["x"], ns["x"])
    assert np.array_equal(data["y"], ns["y"])


def test_save_npz_list_of_dicts_warns_and_roundtrips(tmp_path, caplog):
    """Bug 1b: a ragged list-of-dicts becomes a dtype=object array; warn, load OK."""
    ns = {"results": [{"a": 1}, {"b": 2, "c": 3}]}
    npz = tmp_path / "ragged.npz"

    with caplog.at_level(logging.WARNING, logger="root"):
        info = _save_npz(npz, ns, ["results"])

    assert "results" in caplog.text
    assert "dtype=object" in caplog.text
    assert "allow_pickle=True" in caplog.text
    assert info["results"]["dtype"] == object

    # The replot's loader uses allow_pickle=True; confirm round-trip works.
    data = np.load(npz, allow_pickle=True)
    out = list(data["results"])
    assert out == [{"a": 1}, {"b": 2, "c": 3}]


def test_save_npz_dict_warns_and_replot_uses_item(tmp_path, caplog):
    """Bug 1c: a plain dict becomes a 0-d object array; warn AND emit `.item()`."""
    ns = {"results": {"alpha": 1, "beta": 2}}
    npz = tmp_path / "dict.npz"

    with caplog.at_level(logging.WARNING, logger="root"):
        info = _save_npz(npz, ns, ["results"])

    assert "results" in caplog.text
    assert "dtype=object" in caplog.text
    assert info["results"]["shape"] == ()
    assert info["results"]["dtype"] == object

    _write_replot(tmp_path, "results.keys()", "dict_demo", ["results"], info)
    src = (tmp_path / "replot_dict_demo.py").read_text()

    # New loader passes allow_pickle=True.
    assert "np.load(NPZ, allow_pickle=True)" in src
    # Bare dict gets unwrapped via .item() so users can call .keys() etc.
    assert "results = data['results'].item()" in src


def test_write_replot_rectangular_uses_bare_bind(tmp_path):
    """Sanity: rectangular arrays still bind without `.item()`."""
    info = {"x": {"shape": (5,), "dtype": np.dtype("float64")}}
    _write_replot(tmp_path, "plt.plot(x)", "rect_demo", ["x"], info)
    src = (tmp_path / "replot_rect_demo.py").read_text()
    assert "x = data['x']" in src
    assert ".item()" not in src
    assert "np.load(NPZ, allow_pickle=True)" in src


# ---- Bug 2: cell references variables that were never saved ----


def test_check_free_names_flags_missing_name():
    """Bug 2a: a cell using an unsaved name raises RuntimeError before any I/O."""
    cell = textwrap.dedent("""
        for ax, family in zip(axes.flat, families):
            ax.plot(x_data, results[family])
    """)
    with pytest.raises(RuntimeError) as excinfo:
        _check_free_names(cell, ["x_data", "results"])

    msg = str(excinfo.value)
    # `axes` and `families` are referenced but not provided; `ax` and `family`
    # are bound by the for-loop, `x_data` and `results` are saved.
    assert "'families'" in msg
    assert "'axes'" in msg
    assert "'ax'" not in msg
    assert "'family'" not in msg
    assert "'x_data'" not in msg
    assert "'results'" not in msg


def test_check_free_names_accepts_cell_local_definitions():
    """Bug 2b: helpers, locals, comprehensions, imports defined in the cell are fine."""
    cell = textwrap.dedent("""
        import json
        from math import sqrt as _sqrt

        def _helper(v):
            return v * 2

        scaled = [_helper(v) for v in x_data]
        squares = {k: k * k for k in scaled}
        try:
            payload = json.dumps(squares)
        except TypeError as err:
            payload = str(err)

        fig, ax = plt.subplots()
        ax.plot(x_data, scaled)
        ax.set_title(f"{_sqrt(len(scaled))}: {payload}")
    """)
    # Only x_data is "saved"; everything else is either a builtin, a replot
    # provided name (plt), or defined inside the cell.
    _check_free_names(cell, ["x_data"])


def test_check_free_names_accepts_docstring_example():
    """The example in the mplfreeze docstring must itself pass the Bug 2 check."""
    cell = textwrap.dedent("""
        with plt.style.context("lusca"):
            fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
            axes[0].plot(x_data, sine); axes[0].plot(x_data, cosine)
            axes[1].plot(sine, tanh, linestyle="--")
            axes[1].plot(cosine, tanh, linestyle="--")
            plt.show()
    """)
    _check_free_names(cell, ["x_data", "sine", "cosine", "tanh"])


def test_check_free_names_skips_on_syntax_error(caplog):
    """IPython magics (`!ls`, `%timeit`) make ast.parse fail; we warn and skip."""
    cell = "!ls -la\nplt.plot(x)\n"
    with caplog.at_level(logging.WARNING, logger="root"):
        _check_free_names(cell, [])  # no raise even though `x` is "missing"
    assert "skipping free-name check" in caplog.text


# ---- Replot smoke-test ----


def _bake_run_folder(tmp_path, ns, varnames, cell_src, base="demo"):
    """Build a freeze-folder on disk: NPZ + canonical PNG + replot script."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = _save_npz(tmp_path / f"{base}.npz", ns, varnames)

    exec_ns = {"plt": plt, "np": np, **ns}
    plt.close("all")
    exec(cell_src, exec_ns)
    plt.gcf().savefig(tmp_path / f"{base}.png")
    plt.close("all")

    _write_replot(tmp_path, cell_src, base, varnames, info)
    return base


def test_smoke_test_passes_for_deterministic_cell(tmp_path, caplog):
    """A clean cell whose replot reproduces the figure passes silently."""
    ns = {"x": np.linspace(0, 1, 50), "y": np.sin(np.linspace(0, 1, 50))}
    cell = "fig, ax = plt.subplots(); ax.plot(x, y)"
    base = _bake_run_folder(tmp_path, ns, ["x", "y"], cell)

    with caplog.at_level(logging.INFO, logger="root"):
        _smoke_test_replot(tmp_path, base)

    assert "smoke-test passed" in caplog.text
    assert "FAILED" not in caplog.text
    # Smoke-check artifact is cleaned up.
    assert not (tmp_path / f".{base}.smoke.png").exists()


def test_smoke_test_raises_on_replot_runtime_error(tmp_path):
    """If the cell raises at runtime (e.g. shape mismatch), smoke-test fails loud."""
    ns = {"x": np.linspace(0, 1, 50), "y": np.sin(np.linspace(0, 1, 50))}
    # The cell parses fine and has no free-names, but explodes at runtime.
    cell = "fig, ax = plt.subplots(); ax.plot(x, y); raise ValueError('boom')"

    info = _save_npz(tmp_path / "demo.npz", ns, ["x", "y"])
    # Hand-write the canonical PNG so the smoke-test gets that far.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure().savefig(tmp_path / "demo.png")
    plt.close("all")
    _write_replot(tmp_path, cell, "demo", ["x", "y"], info)

    with pytest.raises(RuntimeError) as excinfo:
        _smoke_test_replot(tmp_path, "demo")
    assert "smoke-test FAILED" in str(excinfo.value)
    assert "ValueError" in str(excinfo.value) or "boom" in str(excinfo.value)


def test_smoke_test_warns_on_pixel_drift(tmp_path, caplog):
    """A cell whose replot draws something different triggers a warning, not a raise."""
    ns = {"x": np.linspace(0, 1, 50)}
    # Save a canonical PNG that the replot will NOT reproduce: replot draws
    # `plt.plot(x)`, but we save a canonical of an empty figure.
    info = _save_npz(tmp_path / "drift.npz", ns, ["x"])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.savefig(tmp_path / "drift.png")  # blank canonical
    plt.close("all")
    _write_replot(
        tmp_path, "fig, ax = plt.subplots(); ax.plot(x, x)", "drift", ["x"], info
    )

    with caplog.at_level(logging.WARNING, logger="root"):
        _smoke_test_replot(tmp_path, "drift")
    assert "not pixel-faithful" in caplog.text


# ---- Environment metadata sidecar ----


def test_write_metadata_records_versions_and_args(tmp_path):
    """Sidecar JSON captures the magic invocation and pinned package versions."""
    import json as _json

    _write_metadata(
        tmp_path,
        base="demo",
        line="demo x y --outdir somewhere",
        varnames=["x", "y"],
        outdir="somewhere",
    )
    meta = _json.loads((tmp_path / "demo.meta.json").read_text())

    assert meta["base"] == "demo"
    assert meta["varnames"] == ["x", "y"]
    assert meta["outdir"] == "somewhere"
    assert meta["magic_line"] == "demo x y --outdir somewhere"
    assert meta["python"].count(".") >= 1
    assert "platform" in meta and meta["platform"]
    assert meta["packages"]["numpy"] is not None
    assert meta["packages"]["matplotlib"] is not None
    assert meta["packages"]["lusca"] is not None
    # freeze_time round-trips as ISO-8601.
    datetime.fromisoformat(meta["freeze_time"])


def test_write_metadata_handles_missing_git(tmp_path, monkeypatch):
    """Outside a git repo, metadata still writes successfully with git=None."""
    import json as _json

    monkeypatch.chdir(tmp_path)  # tmp_path is not a git repo
    _write_metadata(tmp_path, "demo", "demo x", ["x"], "out")
    meta = _json.loads((tmp_path / "demo.meta.json").read_text())
    assert meta["git"] is None


def test_write_metadata_records_git_when_available(tmp_path, monkeypatch):
    """Inside a real git repo we capture commit + branch + dirty flag."""
    import json as _json
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git not installed")

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    # Minimal repo with one commit; -c flags so we don't need a global identity.
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    for cmd in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "commit", "-q", "--allow-empty", "-m", "init"],
    ):
        subprocess.run(cmd, cwd=repo, env=env, check=True, capture_output=True)
    (repo / "dirty.txt").write_text("x")  # working tree is now dirty

    _write_metadata(repo, "demo", "demo x", ["x"], "out")
    meta = _json.loads((repo / "demo.meta.json").read_text())
    assert meta["git"] is not None
    assert len(meta["git"]["commit"]) == 40
    assert meta["git"]["branch"] == "main"
    assert meta["git"]["dirty"] is True
