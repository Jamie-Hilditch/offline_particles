# Configuration file for the Sphinx documentation builder.

# -- Project information
project = "Offline Particles"
copyright = "2026, Jamie Hilditch"
author = "Jamie Hilditch"

# -- General configuration
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# autodoc
autodoc_default_options = {
    "members": False,
    "show-inheritance": True,
    "undoc-members": False,
}
autosummary_generate = True
autosummary_imported_members = False
autosummary_template_dir = "_templates/autosummary"

# Nice formatting
add_module_names = False
python_use_unqualified_type_names = True
toc_object_entries_show_parents = "hide"

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
# suppress_warnings = ["docutils"]
