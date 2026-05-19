"""Kernels for working with buoyant particles."""

import numpy as np

from .._kernels import (
    BoundKernel,
    FieldDataDeclaration,
    ParticleKernel,
    ParticlePropertyDeclaration,
    ScalarDeclaration,
)
from ..input_declarations import (
    STATUS_DECLARATION,
    XIDX_DECLARATION,
    YIDX_DECLARATION,
    ZIDX_DECLARATION,
)
from ..layout_validators import validate_ZYX_ordering
from ._buoyancy_force import buoyancy_force_accumulation

rhs_declaration = ParticlePropertyDeclaration("rhs", np.float64)
rho_property_declaration = ParticlePropertyDeclaration("rho", np.float64)
rho0_declaration = ScalarDeclaration("rho0", np.float64)
g_declaration = ScalarDeclaration("g", np.float64)
rho_field_declaration = FieldDataDeclaration("rho", np.float64, [validate_ZYX_ordering])


def construct_buoyancy_force_accumulation_kernel(
    rhs: str,
    particle_density: str = "rho",
    density_field: str = "rho",
    reference_density: str = "rho0",
    gravity: str = "g",
) -> BoundKernel:
    """Construct a kernel to compute buoyancy force accumulation on particles.

    Parameters
    ----------
    rhs : str
        The name of the particle property to add the computed buoyancy force to.
    particle_density : str, optional
        The binding for the particle property that contains the particle density. Default is "rho".
    density_field : str, optional
        The binding for the field data that contains the fluid density. Default is "rho".
    reference_density : str, optional
        The binding for the scalar that contains the reference fluid density. Default is "rho0".
    gravity : str, optional
        The binding for the scalar that contains the gravitational acceleration. Default is "g".

    Returns
    -------
    BoundKernel
        A bound kernel that computes the buoyancy force on particles.
    """
    kernel = ParticleKernel(
        buoyancy_force_accumulation,
        particle_properties=[
            STATUS_DECLARATION,
            ZIDX_DECLARATION,
            YIDX_DECLARATION,
            XIDX_DECLARATION,
            rhs_declaration,
            rho_property_declaration,
        ],
        scalars=[
            rho0_declaration,
            g_declaration,
        ],
        field_data=[
            rho_field_declaration,
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "rhs": rhs,
            "rho": particle_density,
        },
        scalar_bindings={
            "rho0": reference_density,
            "g": gravity,
        },
        field_data_bindings={
            "rho": density_field,
        },
    )
