import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "nemopy"
author = "nemopy contributors"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_member_order = "bysource"

html_theme = "sphinx_rtd_theme"
html_static_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Pre-import nemopy symbols for all doctest blocks so examples stay concise.
doctest_global_setup = """
from nemopy import (
    _c, _m, mat, eye, as_col, as_mat,
    ColVec, Mat, ShapeError, ConventionWarning,
)
import numpy as np
"""
