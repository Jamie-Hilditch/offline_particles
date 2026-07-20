# Configuration file for the Sphinx documentation builder.

import importlib
import os
import sys
import typing

from sphinx.ext.autodoc._sentinels import ALL

sys.path.insert(0, os.path.abspath("_ext"))

# -- Project information
project = "Offline Particles"
copyright = "2026, Jamie Hilditch"
author = "Jamie Hilditch"

# -- General configuration
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "napoleon_tables",
    "member_labels",
    "member_order",
    "property_types",
    "bases_signature",
]

templates_path = ["_templates"]
exclude_patterns = ["_generated/README.md"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# autodoc
autodoc_default_options = {
    "members": False,
    "undoc-members": True,
}
# Numpydoc-style docstrings document parameter/return types by hand, so drop
# type hints from the rendered signature line rather than duplicating them.
autodoc_typehints = "none"
autosummary_generate = True
autosummary_ignore_module_all = False
autosummary_imported_members = True
autosummary_template_dir = "_templates/autosummary"
# Windows' filesystem is case-insensitive, so the reexported class
# offline_particles.kernels.Status and the submodule offline_particles.kernels.status
# would otherwise generate colliding stub filenames, silently dropping one page.
autosummary_filename_map = {
    "offline_particles.kernels.Status": "offline_particles.kernels.Status_class",
}
# autosummary_context is set below

# Nice formatting
add_module_names = False
python_use_unqualified_type_names = True
toc_object_entries_show_parents = "hide"

# -- Napoleon
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False

# -- Gallery - examples
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "_generated/gallery",
    "filename_pattern": r"^((?!GALLERY_HEADER).)*$",
    "write_computation_times": False,
}

# -- HTML output
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/Jamie-Hilditch/ROMS_particles",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs/source",
    "secondary_sidebar_items": [],
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- warnings
# suppress_warnings = ["docutils"]
suppress_warnings = ["config.cache"]

# -- Extract the value of a type alias for display in the docs
# Type aliases by default only show the alias name (returned by repr) in the docs, their value.


def type_alias_value(fullname: str) -> str | None:
    modname, _, name = fullname.rpartition(".")
    obj = getattr(importlib.import_module(modname), name)
    if isinstance(obj, typing.TypeAliasType):
        return repr(obj.__value__)
    return None  # not a type alias — let autodata render normally


autosummary_context = {"type_alias_value": type_alias_value}

# -- suppress stdlib-inherited members on enum's docs page
#
# Enum classes inherit a pile of int/IntEnum members (bit_length,
# from_bytes, numerator, ...) that :inherited-members: pulls in but that are pure
# stdlib noise.

_ENUM_NAMES = {
    "offline_particles.kernels.status.Status",
    "offline_particles.spatial_arrays.ArrayAxis",
    "offline_particles.spatial_arrays.Stagger",
}


def suppress_enum_inherited_members(app, what, name, obj, options, lines):
    # This event also fires for an unrelated internal call (used to extract
    # the one-line autosummary blurb) where options.inherited_members is None
    # rather than the directive's resolved {"object"} -- only touch the real
    # per-page autoclass directive's options, since that other call path
    # breaks if options.members is forced away from its own default.
    if what == "class" and name in _ENUM_NAMES and options.inherited_members == {"object"}:
        # document_members() only documents everything (want_all) if
        # options.members is the ALL sentinel or options.inherited_members is
        # truthy; clearing inherited_members without forcing members = ALL
        # would make it fall back to "document nothing" instead of "document
        # the enum's own members".
        options.members = ALL
        options.inherited_members = set()
        options.member_order = "bysource"  # keep enum's own members in source order


def setup(app):
    app.connect("autodoc-process-docstring", suppress_enum_inherited_members)
