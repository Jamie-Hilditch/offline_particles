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
    get_required_particle_property_dtypes,
)
from .status import Status, is_active, is_inactive
from .validation import construct_validation_kernel

__all__ = [
    "input_declarations",
    "layout_validators",
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
