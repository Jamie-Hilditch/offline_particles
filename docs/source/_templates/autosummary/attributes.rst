:html_theme.sidebar_secondary.remove:

{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autodata:: {{ objname }}
{%- set _val = type_alias_value(fullname) %}
{%- if _val %}
   :annotation: = {{ _val }}
{%- endif %}