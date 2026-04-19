{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}

{% if classes %}
Classes
-------

.. autosummary::
   :nosignatures:

{% for item in classes %}
   {{ item }}
{% endfor %}
{% endif %}

{% if modules %}
Submodules
----------

.. autosummary::
   :toctree:
   :recursive:
   :template: autosummary/module.rst

{% for item in modules %}
   {{ item }}
{% endfor %}
{% endif %}