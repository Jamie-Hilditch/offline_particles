"""Show a property's return type even though ``autodoc_typehints = "none"``.

Functions/methods document their parameter and return types by hand in
numpydoc-style docstrings, so ``autodoc_typehints = "none"`` avoids
duplicating them from the signature. Properties, though, tend to have a
single-line docstring with no structured Returns section, so with typehints
off they render with no type information at all.

The property's return-type annotation is already computed unconditionally by
autodoc (``_obj_property_type_annotation``); it's only discarded at render
time in ``sphinx.ext.autodoc._renderer._directive_header_lines``, which has a
single ``autodoc_typehints`` gate shared by every object kind (property,
attribute, data, method, function). There's no public per-object-kind hook
for this -- the ``autodoc-process-signature`` event only rewrites the
args/return-annotation used for a function/method's ``def foo(...) -> T``
line, it never touches the property/attribute ``:type:`` line -- so this
patches the renderer directly, following the same approach as
``member_order.py``. Patched at the point of use
(``sphinx.ext.autodoc._generate``, which imported the function by name) since
that module holds its own reference, not at the point of definition. Verified
against the installed Sphinx (9.1.0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sphinx.ext.autodoc import _generate as _autodoc_generate
from sphinx.ext.autodoc._renderer import _directive_header_lines as _original_directive_header_lines

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sphinx.application import Sphinx


def _directive_header_lines_with_property_types(*, autodoc_typehints: str, props, **kwargs) -> Iterator[str]:
    if props.obj_type == "property":
        autodoc_typehints = "signature"
    return _original_directive_header_lines(autodoc_typehints=autodoc_typehints, props=props, **kwargs)


def setup(app: Sphinx) -> None:
    _autodoc_generate._directive_header_lines = _directive_header_lines_with_property_types
