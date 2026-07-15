"""Reorder ``:member-order: groupwise`` groups to attributes/properties, methods, staticmethods, classmethods.

Sphinx's built-in groupwise order documents classmethods first, then
staticmethods, then methods, then attributes/properties last (see
``sphinx.ext.autodoc._property_types._FunctionDefProperties`` and
``_AssignStatementProperties``) -- the opposite of what reads naturally for
this codebase's dataclass-heavy, property-heavy classes. Autodoc's dynamic
backend has no public hook for customising the groupwise sort key (unlike
the legacy ``Documenter``-subclass backend, which exposes ``sort_members``),
so -- following the same approach as ``napoleon_tables.py`` -- this patches
the two private ``_groupwise_order_key`` properties directly. Verified
against the installed Sphinx (9.1.0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sphinx.ext.autodoc._property_types import (  # ty: ignore[possibly-missing-attribute]
    _AssignStatementProperties,
    _FunctionDefProperties,
)

if TYPE_CHECKING:
    from sphinx.application import Sphinx

# Attributes and properties sort together (alphabetically within the group),
# then methods, then staticmethods, then classmethods.
_ATTRIBUTE_OR_PROPERTY = 30
_METHOD = 31
_STATICMETHOD = 32
_CLASSMETHOD = 33


def _function_def_order_key(self: _FunctionDefProperties) -> int:
    if self.obj_type == "method":
        if self.is_classmethod:
            return _CLASSMETHOD
        if self.is_staticmethod:
            return _STATICMETHOD
        return _METHOD
    if self.obj_type == "property":
        return _ATTRIBUTE_OR_PROPERTY
    return 30  # function, decorator (module-level; order unaffected in practice)


def _assign_statement_order_key(self: _AssignStatementProperties) -> int:
    return 40 if self.obj_type == "data" else _ATTRIBUTE_OR_PROPERTY


def setup(app: Sphinx) -> None:
    _FunctionDefProperties._groupwise_order_key = property(_function_def_order_key)  # ty: ignore[invalid-assignment]
    _AssignStatementProperties._groupwise_order_key = property(_assign_statement_order_key)  # ty: ignore[invalid-assignment]
