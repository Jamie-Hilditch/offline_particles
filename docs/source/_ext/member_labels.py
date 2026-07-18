"""Label plain methods and attributes like Sphinx already labels properties/classmethods.

The Python domain (``sphinx.domains.python``) only emits a keyword prefix
(``property``, ``classmethod``, ``abstractmethod``, ...) when an explicit
modifier option is present on the directive -- a plain ``py:method`` or
``py:attribute`` gets no prefix at all. This overrides those two directives
to add a ``method``/``attribute`` fallback prefix so every member is
consistently labelled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sphinx import addnodes
from sphinx.domains.python import PyAttribute, PyMethod

if TYPE_CHECKING:
    from docutils import nodes
    from sphinx.application import Sphinx


class LabeledPyMethod(PyMethod):
    def get_signature_prefix(self, sig: str) -> list[nodes.Node]:
        prefix = list(super().get_signature_prefix(sig))
        if not prefix:  # no classmethod/staticmethod/abstractmethod/final/async
            prefix = [addnodes.desc_sig_keyword("", "method"), addnodes.desc_sig_space()]
        return prefix


class LabeledPyAttribute(PyAttribute):
    def get_signature_prefix(self, sig: str) -> list[nodes.Node]:
        return [
            *super().get_signature_prefix(sig),
            addnodes.desc_sig_keyword("", "attribute"),
            addnodes.desc_sig_space(),
        ]


def setup(app: Sphinx) -> None:
    app.add_directive_to_domain("py", "method", LabeledPyMethod, override=True)
    app.add_directive_to_domain("py", "attribute", LabeledPyAttribute, override=True)
