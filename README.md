# lusca

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/lusca)](https://pypi.org/project/lusca/)
[![CI](https://github.com/evmckinney9/lusca/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/evmckinney9/lusca/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/badge/linter-ruff-green.svg)](https://github.com/astral-sh/ruff)

```bash
pip install lusca
```

`lusca` is a Python library for creating reproducible matplotlib figures using Jupyter magic commands. You often want to use Jupyter for experiments without rerunning the entire notebook to recreate a plot, and saving data alongside figures is essential for artifact generation and reproducibility.

______
### `%%mplfreeze`

```python
%%mplfreeze <name> [vars ...] [--outdir DIR]
```
- `<name>`: Base name for outputs (folder + files).
- `[vars ...]`: Variable names to save into the NPZ file.
- `[--outdir DIR]`: Parent output directory (default: `docs/figs`).

The magic command:
- Captures the data used in your plots and saves it in a compressed NPZ file.
- Exports your figures in PDF, PNG, and SVG.
- Generates a minimal standalone script that reproduces the figure.
- Snapshots Python/package versions and the git commit into `<name>.meta.json`.
- Statically checks the cell for unsaved free names *before* writing anything, then runs the generated replot in a subprocess to confirm the bundle actually reproduces the figure - if `%%mplfreeze` succeeds, the replot is guaranteed to work.
- Applies `lusca`'s built-in stylesheet.

______
### Example

```python
import matplotlib.pyplot as plt
import numpy as np
import lusca
%load_ext lusca
```

```python
x_data = np.linspace(-10, 10, 100)
sine = np.sin(x_data)
cosine = np.cos(x_data)
```

```python
%%mplfreeze trig_demo x_data sine cosine
with plt.style.context("lusca"):
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.6), sharey=True)
    ax.plot(x_data, sine, label="Sine")
    ax.plot(x_data, cosine, label="Cosine")
    plt.show()
```

An example notebook is available in `src/demo.ipynb`. Generated plots are saved under `docs/figs/`:

```
name_stamp/
    name.npz             # saved variables
    name.pdf             # exported figure
    name.png
    name.svg
    name.meta.json       # python/package versions + git commit
    replot_name.py       # standalone replot script
```

______
> [!NOTE]
> If you are using VS Code, set the workspace root as the default directory for saving figures by adding the following to your `settings.json`. Otherwise, output paths will be relative to the notebook location.
> ```json
> "jupyter.notebookFileRoot": "${workspaceFolder}"
> ```
