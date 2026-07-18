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

{% if attributes %}
Types and Constants
-------------------

.. autosummary::
   :toctree:
   :template: autosummary/attributes.rst

{% for item in attributes %}
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