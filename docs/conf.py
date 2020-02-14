
# -- Path setup --------------------------------------------------------------

import os
import sys
import pathlib

# sys.path.insert(0, os.path.abspath('.'))
root = pathlib.Path(__file__).absolute().parent.parent
os.environ["PYTHONPATH"] = str(root)
sys.path.insert(0, str(root))

import legume

# -- Project information -----------------------------------------------------

project = 'legume'
copyright = '2020'
release = legume.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
]

extlinks = {
    "issue": ("https://github.com/fancompute/legume/issues/%s", "GH"),
    "pull": ("https://github.com/fancompute/legume/pull/%s", "PR"),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


autosummary_generate = True
autodoc_typehints = "none"

napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {"logo_only": True}
html_logo = "_static/legume-logo.png"
html_favicon = "_static/favicon.ico"
