"""Submodule defining particle kernels."""

from . import common_inputs, roms, status, timed_activation, timestepping, validation
from ._kernels import (
    BoundKernel,
    FieldDataDeclaration,
    FieldDataType,
    KernelFunction,
    ParticleKernel,
    ParticlePropertiesType,
    ParticlePropertyDeclaration,
    ScalarDeclaration,
    ScalarsType,
    get_required_particle_property_dtypes,
)
from .status import Status, is_active, is_inactive
from .validation import construct_validation_kernel

__all__ = [
    "common_inputs",
    "roms",
    "status",
    "timed_activation",
    "timestepping",
    "validation",
    "BoundKernel",
    "FieldDataDeclaration",
    "FieldDataType",
    "KernelFunction",
    "ParticleKernel",
    "ParticlePropertiesType",
    "ParticlePropertyDeclaration",
    "ScalarDeclaration",
    "ScalarsType",
    "get_required_particle_property_dtypes",
    "Status",
    "is_active",
    "is_inactive",
    "construct_validation_kernel",
]

# Set __module__ for all public classes to this module for cleaner documentation
_module = __name__
for _obj in [
    BoundKernel,
    FieldDataDeclaration,
    ParticleKernel,
    ParticlePropertyDeclaration,
    ScalarDeclaration,
    get_required_particle_property_dtypes,
]:
    _obj.__module__ = _module

# add types to the module docstring
# It's very hacky but it works
__doc__ += """

Types
~~~~~

.. list-table::
   :header-rows: 0
   :widths: 20 80

"""  # type: ignore
for type_alias in [ParticlePropertiesType, ScalarsType, FieldDataType, KernelFunction]:
    try:
        new_entry = f"   * - :py:data:`{type_alias.__name__}`\n"
        new_entry += f"     - ``{type_alias.__value__}``\n"
        __doc__ += new_entry
    except Exception as e:
        # If there's an error accessing __name__ or __value__, skip adding this type to the docstring
        import warnings

        warnings.warn(f"Warning: Could not add {type_alias} to docstring due to error: {e}")
