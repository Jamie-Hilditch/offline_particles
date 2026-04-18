# Configuration file for the Sphinx documentation builder.

# -- Project information
project = "Offline Particles"
copyright = "2026, Jamie Hilditch"
author = "Jamie Hilditch"

# -- General configuration
extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- AutoAPI
autoapi_dirs = ["../../src/offline_particles"]
autoapi_root = "autoapi"
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_use_implicit_namespaces = True
autoapi_add_toctree_entry = False

# -- Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False

# -- HTML output
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    # ...
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    # ...
}
html_static_path = ["_static"]

# -- warnings
suppress_warnings = ["docutils"]
