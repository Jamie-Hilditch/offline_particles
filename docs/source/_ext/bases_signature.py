"""Append "extends X." to a class's signature line instead of a Bases: line.

Replaces autodoc's built-in "Bases: ..." paragraph (already disabled --
``show-inheritance`` was dropped from both ``autodoc_default_options`` and the
``autoclass`` template) with inline text on the signature line itself, e.g.
``class StaticField(data, *, attrs=None) extends Field.``, showing nothing at
all for classes whose only base is ``object``.

An earlier version of this feature injected the sentence into the docstring
via ``autodoc-process-docstring``, but that made the sentence itself get
picked up as the class's autosummary one-line summary (autosummary grabs the
first line of the processed docstring). Signature text is never touched by
autosummary's summary extraction, so this approach sidesteps that entirely.

Getting a real, clickable cross-reference onto the signature line requires
building actual docutils nodes, not a plain string: ``autodoc-process-signature``
can only return a modified ``(args, return_annotation)`` string pair, and the
Python domain's signature grammar (``py_sig_re``) is anchored at the end --
any literal text trailing the closing paren fails to parse, and Sphinx
silently discards the *entire* signature node tree on a parse failure (no
warning). So instead this overrides ``PyClasslike`` (the domain class shared
by ``py:class``/``py:exception``, same directive ``member_labels.py`` already
overrides for method/attribute prefixes) to append nodes after
``handle_signature`` runs, fed by a synthetic ``:extends:`` directive option.

That option is populated by chaining onto
``sphinx.ext.autodoc._generate._directive_header_lines`` -- the same private
function ``property_types.py`` already patches. Because this extension is
registered *after* ``property_types`` in ``conf.py``, it captures whatever
that name currently points to (not the pristine ``_renderer`` original) so
both patches compose instead of one clobbering the other. Verified against
the installed Sphinx (9.1.0): ``props.bases`` (the raw ``__bases__`` tuple)
is computed unconditionally for every class regardless of ``show_inheritance``,
so it's available here for free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.domains.python import PyClasslike, type_to_xref
from sphinx.ext.autodoc import _generate as _autodoc_generate

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sphinx.addnodes import desc_signature
    from sphinx.application import Sphinx

_PACKAGE_ROOT = "offline_particles"
_NO_LINK_PREFIX = "!"


def _is_private_inpackage(base: type) -> bool:
    module = getattr(base, "__module__", "") or ""
    name = getattr(base, "__name__", "")
    in_package = module == _PACKAGE_ROOT or module.startswith(_PACKAGE_ROOT + ".")
    return in_package and name.startswith("_")


def _base_token(base: type) -> str:
    name = getattr(base, "__name__", "")
    if _is_private_inpackage(base):
        # Not documented anywhere, so a cross-reference would be a dead link.
        return _NO_LINK_PREFIX + name
    qualname = getattr(base, "__qualname__", name)
    module = getattr(base, "__module__", "") or ""
    return f"{module}.{qualname}" if module else qualname


def _extends_option_value(props: Any) -> str | None:
    bases = [base for base in (props.bases or ()) if base is not object]
    if not bases:
        return None
    return ",".join(_base_token(base) for base in bases)


def _append_extends_nodes(signode: desc_signature, extends: str, env: Any) -> None:
    tokens = extends.split(",")
    refs = [
        nodes.Text(token[1:]) if token.startswith(_NO_LINK_PREFIX) else type_to_xref(token, env, suppress_prefix=True)
        for token in tokens
    ]
    signode += addnodes.desc_sig_space()
    signode += nodes.Text("extends")
    signode += addnodes.desc_sig_space()
    if len(refs) == 1:
        signode += refs[0]
    elif len(refs) == 2:
        signode += refs[0]
        signode += nodes.Text(" and ")
        signode += refs[1]
    else:
        for ref in refs[:-1]:
            signode += ref
            signode += nodes.Text(", ")
        signode += nodes.Text("and ")
        signode += refs[-1]
    signode += nodes.Text(".")


class ExtendsPyClasslike(PyClasslike):
    option_spec: ClassVar[dict[str, Any]] = {**PyClasslike.option_spec, "extends": directives.unchanged}

    def handle_signature(self, sig: str, signode: desc_signature) -> tuple[str, str]:
        result = super().handle_signature(sig, signode)
        extends = self.options.get("extends")
        if extends:
            _append_extends_nodes(signode, extends, self.env)
        return result


def setup(app: Sphinx) -> None:
    original = _autodoc_generate._directive_header_lines

    def _directive_header_lines_with_extends(*, props: Any, **kwargs: Any) -> Iterator[str]:
        yield from original(props=props, **kwargs)
        if props.obj_type in {"class", "exception"}:
            value = _extends_option_value(props)
            if value:
                yield f"   :extends: {value}"

    _autodoc_generate._directive_header_lines = _directive_header_lines_with_extends  # ty: ignore[invalid-assignment]
    app.add_directive_to_domain("py", "class", ExtendsPyClasslike, override=True)
    app.add_directive_to_domain("py", "exception", ExtendsPyClasslike, override=True)
