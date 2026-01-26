"""Kernels for implementing horizontal advection in ROMS simulations."""

import numpy as np

from ....spatial_arrays import ACTIVE_STAGGERS
from ..._kernels import FieldDataDeclaration, KernelBinding, ParticleKernel, ParticlePropertyDeclaration
from ...common_inputs import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from .linear_interpolation import (
    horizontal_idx_tendency_from_velocity_field,
    horizontal_idx_tendency_from_velocity_property,
)

output_declaration = ParticlePropertyDeclaration("output", np.float64)
velocity_field_declaration = FieldDataDeclaration(
    "velocity", np.float64, z_staggers=ACTIVE_STAGGERS, y_staggers=ACTIVE_STAGGERS, x_staggers=ACTIVE_STAGGERS
)
velocity_property_declaration = ParticlePropertyDeclaration("velocity", np.float64)


def construct_horizontal_idx_tendency_kernel_from_velocity_field(output: str, velocity_field: str) -> ParticleKernel:
    """Construct a kernel to compute a horizontal index tendency from a velocity field."""
    kernel = ParticleKernel(
        horizontal_idx_tendency_from_velocity_field,
        particle_properties=[
            STATUS_DECLARATION,
            ZIDX_DECLARATION,
            YIDX_DECLARATION,
            XIDX_DECLARATION,
            output_declaration,
        ],
        field_data=[
            velocity_field_declaration,
        ],
    )
    return KernelBinding(
        kernel,
        particle_property_bindings={
            "output": output,
        },
        field_data_bindings={
            "velocity": velocity_field,
        },
    )


def construct_horizontal_idx_tendency_kernel_from_velocity_property(
    output: str, velocity_property: str
) -> ParticleKernel:
    """Construct a kernel to compute a horizontal index tendency from a velocity property."""
    kernel = ParticleKernel(
        horizontal_idx_tendency_from_velocity_property,
        particle_properties=[
            STATUS_DECLARATION,
            ZIDX_DECLARATION,
            YIDX_DECLARATION,
            XIDX_DECLARATION,
            velocity_property_declaration,
            output_declaration,
        ],
    )
    return KernelBinding(
        kernel,
        particle_property_bindings={
            "output": output,
            "velocity": velocity_property,
        },
    )
