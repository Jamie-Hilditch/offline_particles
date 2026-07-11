"""Render Napoleon's structured sections as tables instead of field lists.

Covers every numpydoc section Napoleon renders as a field list: Parameters,
Other Parameters, Keyword Arguments, Receives, Returns, Yields, Raises, and
Warns. Table rows have no header row -- the rubric title above each table
(e.g. "Parameters") already makes the column meaning obvious.

Napoleon (``sphinx.ext.napoleon``) doesn't expose a supported hook for
changing how individual numpydoc sections are rendered, so this patches the
relevant ``GoogleDocstring`` methods directly -- ``NumpyDocstring`` inherits
them unchanged, so patching the base class covers both docstring styles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sphinx.ext.napoleon.docstring import GoogleDocstring
from sphinx.locale import _

if TYPE_CHECKING:
    from collections.abc import Callable

    from sphinx.application import Sphinx

    _SectionParser = Callable[[GoogleDocstring, str], list[str]]

_Field = tuple[str, str, list[str]]


def _strip_empty(lines: list[str]) -> list[str]:
    lines = list(lines)
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _format_type_cell(type_str: str) -> list[str]:
    if not type_str:
        return [""]
    if "`" in type_str:
        return [type_str]
    return [f"*{type_str}*"]


def _format_name_cell(name: str) -> list[str]:
    return [f"**{name}**"] if name else [""]


def _desc_cell(desc: list[str]) -> list[str]:
    return _strip_empty(desc) or [""]


def _list_table_row(columns: list[list[str]]) -> list[str]:
    """Render one row of a `.. list-table::` from per-column line lists.

    Returns
    -------
    list[str]
        The RST lines for this row.
    """
    lines: list[str] = []
    for index, cell_lines in enumerate(columns or [[""]]):
        cell_lines = cell_lines or [""]
        marker = "   * - " if index == 0 else "     - "
        lines.append((marker + cell_lines[0]).rstrip())
        for extra in cell_lines[1:]:
            lines.append(f"       {extra}" if extra else "")
    return lines


def _list_table(widths: list[int], rows: list[list[list[str]]]) -> list[str]:
    lines = [
        ".. list-table::",
        "   :class: napoleon-table",
        f"   :widths: {' '.join(str(width) for width in widths)}",
        "",
    ]
    for row in rows:
        lines.extend(_list_table_row(row))
    lines.append("")
    return lines


def _render_parameter_like_section(title: str, fields: list[_Field]) -> list[str]:
    """Render a Name / Type / Description table.

    Used for Parameters, Other Parameters, Keyword Arguments, and Receives.

    Returns
    -------
    list[str]
        The RST lines for the rendered section, or an empty list if `fields`
        is empty.
    """
    if not fields:
        return []
    rows = [[_format_name_cell(name), _format_type_cell(type_), _desc_cell(desc)] for name, type_, desc in fields]
    return [f".. rubric:: {title}", "", *_list_table([20, 20, 60], rows)]


def _render_return_like_section(title: str, fields: list[_Field]) -> list[str]:
    """Render a Type / Description table.

    Used for Returns and Yields.

    Returns
    -------
    list[str]
        The RST lines for the rendered section, or an empty list if `fields`
        is empty.
    """
    if not fields:
        return []
    rows = [[_format_type_cell(type_), _desc_cell(desc)] for _name, type_, desc in fields]
    return [f".. rubric:: {title}", "", *_list_table([25, 75], rows)]


def _render_error_like_section(title: str, fields: list[_Field]) -> list[str]:
    """Render a Type-or-name / Description table.

    Used for Raises and Warns.

    Returns
    -------
    list[str]
        The RST lines for the rendered section, or an empty list if `fields`
        is empty.
    """
    if not fields:
        return []
    rows = [[_format_type_cell(type_ or name), _desc_cell(desc)] for name, type_, desc in fields]
    return [f".. rubric:: {title}", "", *_list_table([25, 75], rows)]


def _parse_parameters_section(self: GoogleDocstring, section: str) -> list[str]:
    use_param = self._config.napoleon_use_param  # ty: ignore[possibly-missing-attribute]
    fields = self._consume_fields(multiple=True) if use_param else self._consume_fields()
    return _render_parameter_like_section(_("Parameters"), fields)


def _parse_other_parameters_section(self: GoogleDocstring, section: str) -> list[str]:
    use_param = self._config.napoleon_use_param  # ty: ignore[possibly-missing-attribute]
    fields = self._consume_fields(multiple=True) if use_param else self._consume_fields()
    return _render_parameter_like_section(_("Other Parameters"), fields)


def _parse_keyword_arguments_section(self: GoogleDocstring, section: str) -> list[str]:
    fields = self._consume_fields()
    return _render_parameter_like_section(_("Keyword Arguments"), fields)


def _parse_receives_section(self: GoogleDocstring, section: str) -> list[str]:
    use_param = self._config.napoleon_use_param  # ty: ignore[possibly-missing-attribute]
    fields = self._consume_fields(multiple=True) if use_param else self._consume_fields()
    return _render_parameter_like_section(_("Receives"), fields)


def _parse_returns_section(self: GoogleDocstring, section: str) -> list[str]:
    fields = self._consume_returns_section()
    return _render_return_like_section(_("Returns"), fields)


def _parse_yields_section(self: GoogleDocstring, section: str) -> list[str]:
    fields = self._consume_returns_section(preprocess_types=True)
    return _render_return_like_section(_("Yields"), fields)


def _parse_raises_section(self: GoogleDocstring, section: str) -> list[str]:
    fields = self._consume_fields(parse_type=False, prefer_type=True)
    return _render_error_like_section(_("Raises"), fields)


def _parse_warns_section(self: GoogleDocstring, section: str) -> list[str]:
    fields = self._consume_fields()
    return _render_error_like_section(_("Warns"), fields)


def setup(app: Sphinx) -> None:
    GoogleDocstring._parse_parameters_section: _SectionParser = _parse_parameters_section
    GoogleDocstring._parse_other_parameters_section: _SectionParser = _parse_other_parameters_section
    GoogleDocstring._parse_keyword_arguments_section: _SectionParser = _parse_keyword_arguments_section
    GoogleDocstring._parse_receives_section: _SectionParser = _parse_receives_section
    GoogleDocstring._parse_returns_section: _SectionParser = _parse_returns_section
    GoogleDocstring._parse_yields_section: _SectionParser = _parse_yields_section
    GoogleDocstring._parse_raises_section: _SectionParser = _parse_raises_section
    GoogleDocstring._parse_warns_section: _SectionParser = _parse_warns_section
