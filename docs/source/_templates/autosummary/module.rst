{{ fullname.split('.')[-1] | escape | underline }}

.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}

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

{% if classes %}
Classes
-------

.. autosummary::
   :toctree:
   :nosignatures:
   :template: autosummary/class.rst

{% for item in classes %}
   {{ item }}
{% endfor %}
{% endif %}

{% if functions %}
Functions
---------

.. autosummary::
   :toctree:
   :nosignatures:
   :template: autosummary/function.rst

{% for item in functions %}
   {{ item }}
{% endfor %}
{% endif %}