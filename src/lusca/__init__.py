# https://github.com/garrettj403/SciencePlots/blob/master/scienceplots/__init__.py
import glob
import os  # pathlib.Path.walk not available in Python <3.12

import matplotlib.pyplot as plt
from matplotlib import rc_params_from_file

import lusca

# register the bundled stylesheets in the matplotlib style library
styles_path = lusca.__path__[0]

# Read every *.mplstyle under styles_path into plt.style.library. matplotlib 3.11
# deprecated style.core.read_style_directory/update_nested_dict (removal in 3.13);
# this is the public-API equivalent (what read_style_directory did internally).
for folder, _, _ in os.walk(styles_path):
    for path in glob.glob(os.path.join(folder, "*.mplstyle")):
        name = os.path.splitext(os.path.basename(path))[0]
        plt.style.library[name] = rc_params_from_file(path, use_default_template=False)

plt.style.available[:] = sorted(plt.style.library.keys())

# Re-export the magic's IPython hooks so users can do `%load_ext lusca`
# instead of the longer `%load_ext lusca.mpl_freeze`.
from lusca.mpl_freeze import (  # noqa: E402, F401
    load_ipython_extension,
    unload_ipython_extension,
)
