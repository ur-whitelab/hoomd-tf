# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import mock
from copy import copy

sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../htf'))

# get the version string from the codebase
with open('../../htf/version.py') as f:
    lines = f.readlines()
exec(lines[0])
# The full version, including alpha/beta/rc tags
release = __version__

# -- Project information -----------------------------------------------------

project = 'HOOMD-TF'
copyright = '2020, Rainier Barrett, \
Dilnoza Amirkulova, Maghesree Chakraborty, \
Heta Gandhi, Andrew D. White'
author = 'Rainier Barrett, Dilnoza Amirkulova, \
Maghesree Chakraborty, Heta Gandhi, Andrew D. White'



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# mock imports so we can generate docs without installing
autodoc_mock_imports = ['hoomd', 'hoomd.md', 'hoomd.md.nlist', 'hoomd.comm', 'tensorflow','numpy','hoomd._htf']

# define master doc for newer versions of sphinx
master_doc = 'index'
