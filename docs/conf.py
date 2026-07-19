import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "tsreduce"
author = "Fernando Moro"
release = "0.3.0"

autodoc_mock_imports = ["torch", "pyts", "pywt", "tqdm"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/MoroFernando/tsreduce",
    "navbar_end": ["navbar-icon-links"],
    "show_toc_level": 2,
}
html_title = "tsreduce"
html_show_sourcelink = False
