# lusca

![CI](https://github.com/evmckinney9/lusca/actions/workflows/ci.yml/badge.svg?branch=main) ![Python](https://img.shields.io/badge/python-3.12-blue.svg) ![Ruff](https://img.shields.io/badge/linter-ruff-green.svg)

`lusca` is a Python library for creating reproducible matplotlib figures using Jupyter magic commands.
- Captures the data used in your plots and saves it in a compressed NPZ file.
- Automatically exports your figures in multiple useful formats.
- Creates a minimal standalone script that reproduces the figure.
- Leverages `lusca`'s built-in stylesheet.

## 📊 `%%mplfreeze` Command

The `%%mplfreeze` magic command is designed for scientific programming workflows in Jupyter notebooks. Often, you want to use Jupyter for experiments but may not want to rerun the entire notebook to recreate plots. Additionally, saving data and figures is essential for artifact generation and reproducibility.


Once you're satisfied with your plot, add the `%%mplfreeze` command to the cell.
```python
%%mplfreeze <name> [vars ...] [--outdir DIR]
```
- `<name>`: Base name for outputs (folder + files).
- `[vars ...]`: Variable names to save into the NPZ file.
- `[--outdir DIR]`: (Optional) Parent output directory (default: `docs/figs`).

### Example

```python
# Example:
%%mplfreeze trig_demo x_data sine cosine tanh
with plt.style.context("lusca"):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    axes[0].plot(x_data, sine, label="Sine")
    axes[0].plot(x_data, cosine, label="Cosine")
    plt.show()
```

An example notebook is available in `src/lusca/01_main.ipynb`. The generated plots are saved in `docs/figs/` with the following structure:

```
name_stamp/
    name.npz
    name.pdf
    name.png
    name.svg
    replot_name.py
```
## Installation

Install `lusca` directly from GitHub:

```bash
pip install -e git+https://github.com/evmckinney9/lusca#egg=lusca
```
## 👯 Contributors

<a href="https://github.com/evmckinney9/lusca/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=evmckinney9/lusca"/>
</a>
