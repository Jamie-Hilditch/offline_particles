"""Submodule defining particle kernels.

Types
~~~~~

.. list-table::
   :header-rows: 0
   :widths: 20 80

   * - :py:data:`ParticlePropertiesType`
     - ``Mapping[str, npt.NDArray]``
   * - :py:data:`ScalarsType`
     - ``Mapping[str, np.generic]``
   * - :py:data:`FieldDataType`
     - ``Mapping[str, FieldData]`` where the ``FieldData`` are instances of :py:class:`~offline_particles.fields.FieldData`
   * - :py:data:`KernelFunction`
     - ``Callable[[ParticlePropertiesType, ScalarsType, FieldDataType], None]``
"""

from . import input_declarations, layout_validators, roms, status, timed_activation, timestepping, validation
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
    kernel_function,
)
from .status import Status, is_active, is_inactive
from .validation import construct_validation_kernel_from_bbox

__all__ = [
    "BoundKernel",
    "FieldDataDeclaration",
    "FieldDataType",
    "KernelFunction",
    "ParticleKernel",
    "ParticlePropertiesType",
    "ParticlePropertyDeclaration",
    "ScalarDeclaration",
    "ScalarsType",
    "Status",
    "construct_validation_kernel_from_bbox",
    "input_declarations",
    "is_active",
    "is_inactive",
    "kernel_function",
    "layout_validators",
    "roms",
    "status",
    "timed_activation",
    "timestepping",
    "validation",
]
