"""Kernels for working with buoyant particles."""

import numpy as np

from ...spatial_arrays import ACTIVE_STAGGERS
from .._kernels import (
    BoundKernel,
    FieldDataDeclaration,
    ParticleKernel,
    ParticlePropertyDeclaration,
    ScalarDeclaration,
)
from ..common_inputs import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from ._buoyancy_force import buoyancy_force_accumulation

rhs_declaration = ParticlePropertyDeclaration("rhs", np.float64)
rho_property_declaration = ParticlePropertyDeclaration("rho", np.float64)
rho0_declaration = ScalarDeclaration("rho0", np.float64)
g_declaration = ScalarDeclaration("g", np.float64)
rho_field_declaration = FieldDataDeclaration(
    "rho", np.float64, z_staggers=ACTIVE_STAGGERS, y_staggers=ACTIVE_STAGGERS, x_staggers=ACTIVE_STAGGERS
)


def construct_buoyancy_force_accumulation_kernel(
    rhs: str,
    particle_density: str = "rho",
    density_field: str = "rho",
    reference_density: str = "rho0",
    gravity: str = "g",
) -> BoundKernel:
    """Construct a kernel to compute buoyancy force accumulation on particles."""
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
