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
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False

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

# -- warnings
# suppress_warnings = ["docutils"]


# -- __modules__ overrides for cleaner documentation
#
# The following modules contain classes and functions that are defined in private implementation modules
# but we want them to appear as if they are defined in the public modules for cleaner documentation.
# Therefore, we override the __module__ for these objects.
# However, since Sphinx imports the modules multiple times during the build process, we need to set __module__
# in a way that works regardless of the import order. We can do this by connecting to the "builder-inited" event in
# Sphinx and setting __module__ for the relevant objects at that time.
def apply_events_module_overrides(app):
    import offline_particles.events as events_module
    from offline_particles.events import (
        AtIterationScheduler,
        AtTimeScheduler,
        Event,
        IterationSchedulerProtocol,
        RecurringIterationScheduler,
        RecurringTimeScheduler,
        SimulationState,
        TimeSchedulerProtocol,
    )

    _module = events_module.__name__
    for _obj in [
        AtIterationScheduler,
        AtTimeScheduler,
        Event,
        IterationSchedulerProtocol,
        RecurringIterationScheduler,
        RecurringTimeScheduler,
        SimulationState,
        TimeSchedulerProtocol,
    ]:
        _obj.__module__ = _module


def apply_kernels_module_overrides(app):
    import offline_particles.kernels as kernels_module
    from offline_particles.kernels import (
        BoundKernel,
        FieldDataDeclaration,
        ParticleKernel,
        ParticlePropertyDeclaration,
        ScalarDeclaration,
    )

    _module = kernels_module.__name__
    for _obj in [
        BoundKernel,
        FieldDataDeclaration,
        ParticleKernel,
        ParticlePropertyDeclaration,
        ScalarDeclaration,
    ]:
        _obj.__module__ = _module


def apply_output_module_overrides(app):
    import offline_particles.output as output_module
    from offline_particles.output import (
        AbstractOutputWriter,
        AbstractOutputWriterBuilder,
        Output,
        ZarrOutputBuilder,
        ZarrOutputWriter,
        interpolate_fields,
    )

    _module = output_module.__name__
    for _obj in [
        AbstractOutputWriter,
        AbstractOutputWriterBuilder,
        Output,
        ZarrOutputBuilder,
        ZarrOutputWriter,
        interpolate_fields,
    ]:
        _obj.__module__ = _module


def setup(app):
    app.connect("builder-inited", apply_events_module_overrides, priority=0)
    app.connect("builder-inited", apply_kernels_module_overrides, priority=0)
    app.connect("builder-inited", apply_output_module_overrides, priority=0)
