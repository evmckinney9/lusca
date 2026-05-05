"""Tests for src.lusca.mpl_freeze."""

import logging
import textwrap

import numpy as np
import pytest

from src.lusca.mpl_freeze import (
    _check_free_names,
    _parse_line,
    _save_npz,
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
