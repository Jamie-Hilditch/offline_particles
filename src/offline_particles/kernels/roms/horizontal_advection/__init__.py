"""Kernels for implementing horizontal advection in ROMS simulations."""

import numpy as np

from ....spatial_arrays import ACTIVE_STAGGERS, INACTIVE_STAGGERS
from ..._kernels import BoundKernel, FieldDataDeclaration, ParticleKernel, ParticlePropertyDeclaration
from ...common_inputs import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from .linear_interpolation import (
    horizontal_idx_tendency_from_velocity_field,
    horizontal_idx_tendency_from_velocity_property,
)

didx_declaration = ParticlePropertyDeclaration("didx", np.float64)
velocity_field_declaration = FieldDataDeclaration(
    "velocity", np.float64, z_staggers=ACTIVE_STAGGERS, y_staggers=ACTIVE_STAGGERS, x_staggers=ACTIVE_STAGGERS
)
grid_spacing_declaration = FieldDataDeclaration(
    "grid_spacing", np.float64, z_staggers=INACTIVE_STAGGERS, y_staggers=ACTIVE_STAGGERS, x_staggers=ACTIVE_STAGGERS
)
vel_property_declaration = ParticlePropertyDeclaration("vel", np.float64)


def construct_horizontal_idx_tendency_kernel_from_velocity_field(
    didx: str, velocity_field: str, grid_spacing_field: str
) -> BoundKernel:
    """Construct a kernel to compute a horizontal index tendency from a velocity field.

    Parameters
    ----------
    didx : str
        Binding for the particle property to store the horizontal index tendency.
    velocity_field : str
        Binding for the velocity field data.
    grid_spacing_field : str
        Binding for the grid spacing field data.

    Returns
    -------
    BoundKernel
        The constructed kernel.
    """
    kernel = ParticleKernel(
        horizontal_idx_tendency_from_velocity_field,
        particle_properties=[
            STATUS_DECLARATION,
            ZIDX_DECLARATION,
            YIDX_DECLARATION,
            XIDX_DECLARATION,
            didx_declaration,
        ],
        field_data=[
            velocity_field_declaration,
            grid_spacing_declaration,
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            didx_declaration.name: didx,
        },
        field_data_bindings={
            velocity_field_declaration.name: velocity_field,
            grid_spacing_declaration.name: grid_spacing_field,
        },
    )


def construct_horizontal_idx_tendency_kernel_from_velocity_property(
    didx: str, velocity_property: str, grid_spacing_field: str
) -> BoundKernel:
    """Construct a kernel to compute a horizontal index tendency from a velocity property.

    Parameters
    ----------
    didx : str
        Binding for the particle property to store the horizontal index tendency.
    velocity_property : str
        Binding for the velocity particle property.
    grid_spacing_field : str
        Binding for the grid spacing field data.
    """
    kernel = ParticleKernel(
        horizontal_idx_tendency_from_velocity_property,
        particle_properties=[
            STATUS_DECLARATION,
            ZIDX_DECLARATION,
            YIDX_DECLARATION,
            XIDX_DECLARATION,
            vel_property_declaration,
            didx_declaration,
        ],
        field_data=[grid_spacing_declaration],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            didx_declaration.name: didx,
            vel_property_declaration.name: velocity_property,
        },
        field_data_bindings={
            grid_spacing_declaration.name: grid_spacing_field,
        },
    )
