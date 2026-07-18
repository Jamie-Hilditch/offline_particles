"""Submodule defining particle kernels."""

from . import (
    advection,
    base,
    buoyancy,
    input_declarations,
    interpolation,
    layout_validators,
    relaxation,
    roms,
    status,
    timed_activation,
    timestepping,
    validation,
)
from ._kernels import (
    BoundKernel,
    FieldDataDeclaration,
    FieldDataType,
    KernelFunction,
    KernelInputDeclaration,
    LayoutValidator,
    ParticleKernel,
    ParticlePropertiesType,
    ParticlePropertyDeclaration,
    ScalarDeclaration,
    ScalarsType,
    kernel_function,
)

# from .status import Status, is_active, is_inactive
# from .validation import construct_validation_kernel_from_bbox

__all__ = [
    "BoundKernel",
    "FieldDataDeclaration",
    "FieldDataType",
    "KernelFunction",
    "KernelInputDeclaration",
    "LayoutValidator",
    "ParticleKernel",
    "ParticlePropertiesType",
    "ParticlePropertyDeclaration",
    "ScalarDeclaration",
    "ScalarsType",
    "advection",
    "base",
    "buoyancy",
    "input_declarations",
    "interpolation",
    "kernel_function",
    "layout_validators",
    "relaxation",
    "roms",
    "status",
    "timed_activation",
    "timestepping",
    "validation",
]

# Public type docstrings
# This is a bit of a hack becuase sphinx can only pick up attribute docstrings in the file
# where the attribute is defined. These type aliases are defined in _kernels.py and then
# redefined here to provide docstrings for them in the public API documentation.

#: The type of the particle properties input to a kernel function.
ParticlePropertiesType = ParticlePropertiesType  # noqa: PLW0127
#: The type of the scalar inputs to a kernel function.
ScalarsType = ScalarsType  # noqa: PLW0127
#: The type of the field data inputs to a kernel function.
FieldDataType = FieldDataType  # noqa: PLW0127
#: The type signature of functions called by a :class:`ParticleKernel`.
KernelFunction = KernelFunction  # noqa: PLW0127
#: The type signature of a layout validator.
LayoutValidator = LayoutValidator  # noqa: PLW0127
