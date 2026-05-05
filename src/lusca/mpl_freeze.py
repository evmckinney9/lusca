"""Jupyter magic for freezing matplotlib plots and saving data.

This module provides the %%mplfreeze magic command for Jupyter/IPython, which captures
plotting cells, saves specified variables to compressed NPZ files, exports figures in
multiple formats, and generates standalone replot scripts for reproducibility.
"""

from __future__ import annotations

import argparse
import ast
import builtins as _builtins
import json
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

# Names the generated replot script binds before executing the captured cell.
_REPLOT_PROVIDED: frozenset[str] = frozenset(
    {"np", "plt", "lusca", "os", "Path", "data"}
)


# ---- parse: %%mplfreeze <name> [vars ...] [--outdir DIR] ----
def _parse_line(line: str):
    p = argparse.ArgumentParser(prog="%%mplfreeze", add_help=False)
    p.add_argument("name", help="Base name for outputs (folder + files)")
    p.add_argument("vars", nargs="*", help="Variable names to save into the NPZ")
    p.add_argument("--outdir", default="docs/figs", help="Parent output directory")
    a = p.parse_args(shlex.split(line))
    return a.name, a.vars, a.outdir


def _warn_on_reserved_varnames(varnames: list[str]) -> None:
    """Warn if any saved varname shadows a name the replot pre-binds.

    The replot does ``import numpy as np`` etc. before binding NPZ data,
    so a saved variable named ``np`` will silently overwrite the numpy
    import in the replot's namespace. Almost always a mistake.
    """
    clashes = sorted(set(varnames) & _REPLOT_PROVIDED)
    if clashes:
        logging.warning(
            f"[mplfreeze] saved variable(s) {clashes} shadow names the replot "
            f"pre-binds {sorted(_REPLOT_PROVIDED)}; the NPZ value will overwrite "
            f"the import (e.g. saving 'np' replaces numpy). Rename if "
            f"unintentional."
        )


def _warn_on_extra_figures(fignums: list[int], kept_num: int) -> None:
    """Warn if the cell created multiple figures; only ``kept_num`` is saved."""
    if len(fignums) > 1:
        logging.warning(
            f"[mplfreeze] cell created {len(fignums)} figures "
            f"(numbers={sorted(fignums)}); only fig#{kept_num} was saved. "
            f"To capture the others, split them into separate %%mplfreeze "
            f"cells or assign the desired one to `fig`."
        )


def _collect_loaded_names(tree: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _collect_cell_defined_names(tree: ast.AST) -> set[str]:
    """Best-effort set of names bound anywhere in the cell.

    Conservative: walks the whole tree and treats any binding (assignment,
    import, function/class/lambda parameter, comprehension target, except
    handler, walrus) as cell-scope. Star-imports cannot be enumerated; we
    insert the sentinel "*" so the caller can warn.
    """
    defined: set[str] = set()

    def _add_target(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            defined.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                _add_target(elt)
        elif isinstance(node, ast.Starred):
            _add_target(node.value)
        # Attribute / Subscript targets bind nothing new.

    def _add_args(args: ast.arguments) -> None:
        for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            defined.add(a.arg)
        if args.vararg:
            defined.add(args.vararg.arg)
        if args.kwarg:
            defined.add(args.kwarg.arg)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                _add_target(t)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            _add_target(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            _add_target(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    _add_target(item.optional_vars)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            _add_args(node.args)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Lambda):
            _add_args(node.args)
        elif isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            for gen in node.generators:
                _add_target(gen.target)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    defined.add("*")
                else:
                    defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                defined.add(node.name)
        elif isinstance(node, ast.NamedExpr):
            _add_target(node.target)

    return defined


def _check_free_names(cell_src: str, varnames: list[str]) -> None:
    """Raise RuntimeError if the cell references names not bound at replot time.

    The replot script binds: numpy as np, pyplot as plt, lusca, os, Path,
    data, plus each saved variable. Anything else used in the cell must
    either be a builtin or be defined inside the cell itself.
    """
    try:
        tree = ast.parse(cell_src)
    except SyntaxError:
        logging.warning(
            "[mplfreeze] could not parse captured cell as Python (likely "
            "contains IPython magics or shell escapes); skipping free-name "
            "check."
        )
        return

    defined_in_cell = _collect_cell_defined_names(tree)
    if "*" in defined_in_cell:
        logging.warning(
            "[mplfreeze] captured cell uses `from ... import *`; free-name "
            "check cannot enumerate names provided by the star import."
        )
        defined_in_cell.discard("*")

    loaded = _collect_loaded_names(tree)
    provided = (
        set(varnames) | set(_REPLOT_PROVIDED) | set(dir(_builtins)) | defined_in_cell
    )
    missing = sorted(loaded - provided)
    if missing:
        raise RuntimeError(
            f"[mplfreeze] captured cell references unsaved free names: "
            f"{missing}. Add these to the %%mplfreeze line, or inline their "
            f"values inside the cell."
        )


def _save_npz(path: Path, ns: dict, varnames: list[str]) -> dict[str, dict]:
    """Save varnames from ns into a compressed NPZ; return per-variable metadata.

    Emits a logging.warning for any variable whose ``np.asarray`` coercion
    yields ``dtype=object`` — that artifact will require ``allow_pickle=True``
    to load and is brittle across NumPy/Python versions.
    """
    arrays: dict[str, np.ndarray] = {}
    info: dict[str, dict] = {}
    for v in varnames:
        if v not in ns:
            raise RuntimeError(
                f"[mplfreeze] Variable '{v}' not found in the notebook namespace."
            )
        arr = np.asarray(ns[v])
        arrays[v] = arr
        info[v] = {"shape": arr.shape, "dtype": arr.dtype}
        if arr.dtype == object:
            logging.warning(
                f"[mplfreeze] Variable {v!r} coerced to a dtype=object array "
                f"(shape={arr.shape}); the saved .npz will use pickle and "
                f"requires allow_pickle=True to load. Pickled .npz is brittle "
                f"across NumPy/Python versions — consider flattening {v!r} "
                f"into rectangular numeric arrays for long-term reproducibility."
            )
    if not arrays:
        raise RuntimeError(
            "[mplfreeze] No variables provided. Use: %%mplfreeze name x y ..."
        )
    np.savez_compressed(path, **arrays)
    return info


def _write_replot(
    root: Path,
    cell_src: str,
    base: str,
    varnames: list[str],
    info: dict[str, dict],
) -> None:
    bind_lines = []
    for v in varnames:
        meta = info[v]
        if meta["dtype"] == object and meta["shape"] == ():
            # 0-d object array — unwrap so users get the original Python value.
            bind_lines.append(f"        {v} = data[{v!r}].item()")
        else:
            bind_lines.append(f"        {v} = data[{v!r}]")
    binds = "\n".join(bind_lines)
    code = f'''# replot_{base}.py — auto-generated by %%mplfreeze
import os
import sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt

try:
    import lusca  # noqa: F401  — registers bundled matplotlib stylesheets
except ImportError:
    pass

HERE = Path(__file__).parent
NPZ  = HERE / "{base}.npz"

def main(out_path=None):
    _prev_cwd = os.getcwd()
    os.chdir(HERE)
    try:
        data = np.load(NPZ, allow_pickle=True)
{binds}

        # ---- begin captured plotting cell ----
{textwrap.indent(cell_src.strip(), "        ")}
        # ---- end captured plotting cell ----

        fig = plt.gcf()
        if out_path:
            fig.savefig(out_path)
        return fig
    finally:
        os.chdir(_prev_cwd)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
'''
    (root / f"replot_{base}.py").write_text(code)


def _git_snapshot(cwd: Path) -> dict | None:
    """Return {commit, branch, dirty} for the git repo at cwd, or None."""

    def _git(*args: str) -> str | None:
        try:
            r = subprocess.run(
                ["git", "-C", str(cwd), *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _write_metadata(
    root: Path,
    base: str,
    line: str,
    varnames: list[str],
    outdir: str,
) -> None:
    """Write `<base>.meta.json` snapshotting the freeze environment.

    Captures Python/numpy/matplotlib/lusca versions, the platform string,
    git commit + dirty state (if cwd is in a repo), and the magic invocation.
    Future-you needs this when matplotlib's defaults shift and the frozen
    PNG no longer matches what the replot draws.
    """
    meta = {
        "freeze_time": datetime.now().isoformat(timespec="seconds"),
        "magic_line": line,
        "base": base,
        "varnames": list(varnames),
        "outdir": outdir,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": _package_version("numpy"),
            "matplotlib": _package_version("matplotlib"),
            "lusca": _package_version("lusca"),
        },
        "git": _git_snapshot(Path.cwd()),
    }
    (root / f"{base}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _maybe_dedupe_against_latest(outdir: Path, base: str, root: Path) -> Path:
    """If ``{base}_latest`` already points at an identical run, delete root.

    "Identical" means the NPZ and the generated replot script are byte-equal
    (the figure exports may differ in embedded timestamps even for matching
    inputs, so we don't compare them). When the user re-runs an unchanged
    cell, this prevents the output directory from accumulating one new
    timestamped folder per execution.

    Returns the run folder that should be considered the "current" one —
    either the prior one (if dedup happened) or ``root`` unchanged.
    """
    latest = outdir / f"{base}_latest"
    if not latest.is_symlink():
        return root
    try:
        prior = (outdir / os.readlink(latest)).resolve()
    except OSError:
        return root
    if not prior.is_dir() or prior == root.resolve():
        return root
    npz_a, npz_b = root / f"{base}.npz", prior / f"{base}.npz"
    rep_a, rep_b = root / f"replot_{base}.py", prior / f"replot_{base}.py"
    if not (npz_b.exists() and rep_b.exists()):
        return root
    if (
        npz_a.read_bytes() == npz_b.read_bytes()
        and rep_a.read_bytes() == rep_b.read_bytes()
    ):
        shutil.rmtree(root)
        logging.info(
            f"[mplfreeze] new run is identical to {prior.name}; removed "
            f"redundant {root.name} and kept the existing folder."
        )
        return prior
    return root


def _update_latest_symlink(outdir: Path, base: str, target: Path) -> None:
    """Point ``{base}_latest`` at ``target`` (a sibling folder).

    Falls back to writing ``{base}_latest.txt`` containing the target name
    on platforms / filesystems where symlinks are unavailable (notably
    Windows without developer mode).
    """
    link = outdir / f"{base}_latest"
    target_name = target.name
    if link.is_symlink() or link.exists():
        try:
            link.unlink()
        except OSError:
            pass
    try:
        link.symlink_to(target_name, target_is_directory=True)
    except OSError as e:
        logging.warning(
            f"[mplfreeze] could not create symlink {link.name} → "
            f"{target_name} ({e}); writing {base}_latest.txt pointer instead."
        )
        (outdir / f"{base}_latest.txt").write_text(target_name + "\n")


def _smoke_test_replot(root: Path, base: str, pixel_threshold: float = 0.01) -> None:
    """Run the generated replot in a subprocess and verify it reproduces the figure.

    Raises RuntimeError if the replot fails to execute or fails to produce a
    figure file. Logs a warning if the replotted image differs from the
    canonical PNG by more than ``pixel_threshold`` (matplotlib loads PNGs as
    floats in [0, 1], so 0.01 is a generous tolerance for cosmetic drift).
    """
    import matplotlib.image as mpimg

    # Absolute paths matter: the replot subprocess chdir's to its own folder
    # before resolving any path it received, so a relative check_png would
    # be interpreted relative to the new cwd and fail.
    root = root.resolve()
    replot = root / f"replot_{base}.py"
    canonical_png = root / f"{base}.png"
    check_png = root / f".{base}.smoke.png"
    check_png.unlink(missing_ok=True)

    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        [sys.executable, str(replot), str(check_png)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"[mplfreeze] replot smoke-test FAILED: {replot.name} exited "
            f"with code {proc.returncode}. The frozen run is NOT reproducible "
            f"as-is.\n--- replot stderr ---\n{proc.stderr.rstrip()}"
        )
    if not check_png.exists():
        raise RuntimeError(
            f"[mplfreeze] replot smoke-test ran but produced no figure. The "
            f"captured cell may close all figures before main() returns."
        )

    try:
        canon = mpimg.imread(canonical_png)
        check = mpimg.imread(check_png)
    finally:
        check_png.unlink(missing_ok=True)

    if canon.shape != check.shape:
        raise RuntimeError(
            f"[mplfreeze] replot smoke-test: figure shape {check.shape} does "
            f"not match canonical {canon.shape}."
        )
    max_diff = float(np.abs(canon.astype(float) - check.astype(float)).max())
    if max_diff > pixel_threshold:
        logging.warning(
            f"[mplfreeze] replot smoke-test: image differs from canonical by "
            f"max_diff={max_diff:.4f} (>{pixel_threshold}). Replot runs but "
            f"the figure is not pixel-faithful — likely non-deterministic "
            f"content in the cell (timestamps, unseeded RNG, etc.)."
        )
    else:
        logging.info(f"[mplfreeze] replot smoke-test passed (max_diff={max_diff:.6f}).")


def mplfreeze(line: str, cell: str):
    """Freeze a matplotlib cell into a reproducible artifact bundle.

    Captures the cell, saves the named variables into a compressed NPZ,
    exports the figure as PDF/SVG/PNG, writes a standalone
    ``replot_<name>.py`` that regenerates the figure from the NPZ,
    snapshots the env (Python + package versions + git commit) into
    ``<name>.meta.json``, then *runs* the generated replot in a
    subprocess to confirm the bundle reproduces the figure before
    declaring success.

    Static checks before any I/O:
        * Saved variable names that shadow replot-bound names
          (``np``, ``plt``, ``lusca``, ``os``, ``Path``, ``data``) → warning.
        * Free names referenced in the cell that aren't saved or
          defined in-cell → RuntimeError.

    Runtime checks after the cell executes:
        * Multiple open figures (only ``plt.gcf()`` is saved) → warning.
        * Replot subprocess exit ≠ 0 or missing figure → RuntimeError.
        * Replotted PNG differs from the canonical by > 0.01 → warning
          (not an error; non-determinism like timestamps or unseeded RNG
          is the most common cause).

    Args:
        line: ``"name var1 var2 ... [--outdir DIR]"``. Default outdir
            is ``docs/figs``.
        cell: Python source of the cell to capture and execute.

    Example:
        %%mplfreeze trig_demo x_data sine cosine tanh
        with plt.style.context("lusca"):
            fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
            axes[0].plot(x_data, sine); axes[0].plot(x_data, cosine)
            axes[1].plot(sine, tanh, linestyle="--")
            axes[1].plot(cosine, tanh, linestyle="--")

    Output layout under ``<outdir>/<name>_<timestamp>/``::

        <name>.npz             # saved variables
        <name>.{pdf,svg,png}   # exported figure
        <name>.meta.json       # env + git snapshot
        replot_<name>.py       # standalone replot script

    Raises:
        RuntimeError: not in IPython/Jupyter; no figure produced; cell
            references unsaved free names; or the replot smoke-test
            fails (subprocess non-zero exit, missing figure, shape
            mismatch).
    """
    import matplotlib.pyplot as plt
    from IPython import get_ipython
    from matplotlib.figure import Figure

    ip = get_ipython()
    if ip is None:
        raise RuntimeError("%%mplfreeze must run inside IPython/Jupyter.")
    ns = ip.user_ns

    base, varnames, outdir = _parse_line(line)

    # Validate the captured cell can run standalone before touching the disk.
    _warn_on_reserved_varnames(varnames)
    _check_free_names(cell, varnames)

    # create run folder
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(outdir) / f"{base}_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    # save arrays
    info = _save_npz(root / f"{base}.npz", ns, varnames)
    logging.info(f"Saved {len(varnames)} arrays → {root / f'{base}.npz'}")

    # run the plotting cell now
    ip.run_cell(cell)

    # snapshot the single figure
    fig = ns.get("fig", None)
    if not isinstance(fig, Figure):
        fig = plt.gcf()
    if not isinstance(fig, Figure):
        raise RuntimeError(
            "[mplfreeze] No Matplotlib Figure found as 'fig' or current figure."
        )
    _warn_on_extra_figures(plt.get_fignums(), fig.number)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(root / f"{base}.{ext}")
    logging.info(f"Saved figure → {root}/{base}.{{pdf,svg,png}}")

    # write replot script with explicit local bindings
    _write_replot(root, cell, base, varnames, info)
    logging.info(f"Wrote {root / f'replot_{base}.py'}")

    # snapshot environment versions / git state for future debugging
    _write_metadata(root, base, line, varnames, outdir)
    logging.info(f"Wrote {root / f'{base}.meta.json'}")

    # Smoke-test: actually exec the replot and confirm it produces the same
    # figure. If freeze succeeds, the replot is *guaranteed* to work later.
    _smoke_test_replot(root, base)

    # Dedupe vs. the prior `{base}_latest` if the new run is identical, then
    # update the symlink so callers can embed a stable path in their docs.
    outdir_path = Path(outdir)
    final_root = _maybe_dedupe_against_latest(outdir_path, base, root)
    _update_latest_symlink(outdir_path, base, final_root)

    logging.info(f"Run folder: {final_root} (latest → {base}_latest)")


# ---- IPython extension hooks ----
def load_ipython_extension(ip):
    """Load the mplfreeze magic command into IPython.

    Args:
        ip: The IPython instance to register the magic command with.
    """
    mgr = ip.magics_manager.magics
    if "cell" in mgr and "mplfreeze" in mgr["cell"]:
        del mgr["cell"]["mplfreeze"]
    ip.register_magic_function(mplfreeze, magic_kind="cell", magic_name="mplfreeze")


def unload_ipython_extension(ip):
    """Unload the mplfreeze magic command from IPython.

    Args:
        ip: The IPython instance to unregister the magic command from.
    """
    mgr = ip.magics_manager.magics
    if "cell" in mgr and "mplfreeze" in mgr["cell"]:
        del mgr["cell"]["mplfreeze"]
