"""Enforce that ``@property`` docstrings are a one-line summary only.

See the "@property docstrings" rule in AGENTS.md: if a property needs more than a
one-line summary, it should be a method instead.
"""

import importlib
import inspect
import pkgutil

import pytest

import offline_particles


def _discover_public_properties() -> list[tuple[str, property]]:
    properties = []
    for module_info in pkgutil.walk_packages(offline_particles.__path__, prefix="offline_particles."):
        module = importlib.import_module(module_info.name)
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            for name, member in vars(cls).items():
                if isinstance(member, property) and not name.startswith("_"):
                    qualname = f"{module.__name__}.{cls_name}.{name}"
                    properties.append((qualname, member))
    return sorted(properties, key=lambda item: item[0])


PROPERTIES = _discover_public_properties()


@pytest.mark.parametrize("qualname,prop", PROPERTIES, ids=[qualname for qualname, _ in PROPERTIES])
def test_property_docstring_is_one_line_summary(qualname: str, prop: property) -> None:
    doc = prop.__doc__
    if doc is None:
        pytest.skip(f"{qualname} has no docstring (not required by D1)")
    else:
        assert len(inspect.cleandoc(doc).splitlines()) == 1, (
            f"{qualname} docstring must be a one-line summary; if it needs more, make it a method instead"
        )
