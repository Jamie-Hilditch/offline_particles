"""Kernels for working with ROMS vertical coordinates."""

import numpy as np

from ..._kernels import (
    BoundKernel,
    FieldDataDeclaration,
    ParticleKernel,
    ParticlePropertyDeclaration,
    ScalarDeclaration,
)
from ...input_declarations import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from ...layout_validators import validate_YX_ordering, validate_Z_ordering
from .vertical_coordinate import compute_z_kernel_function, compute_zidx_kernel_function

__all__ = [
    "construct_compute_z_kernel_function",
    "construct_compute_zidx_kernel_function",
]

z_declaration = ParticlePropertyDeclaration("z", np.float64)
hc_declaration = ScalarDeclaration("hc", np.float64)
NZ_declaration = ScalarDeclaration("NZ", np.int32)
h_declaration = FieldDataDeclaration("h", np.float64, [validate_YX_ordering])
zeta_declaration = FieldDataDeclaration("zeta", np.float64, [validate_YX_ordering])
C_declaration = FieldDataDeclaration("C", np.float64, [validate_Z_ordering])


def construct_compute_z_kernel(
    z: str = "z",
    hc: str = "hc",
    NZ: str = "NZ",
    h: str = "h",
    zeta: str = "zeta",
    C: str = "C",
) -> BoundKernel:
    """Construct a kernel to compute the physical vertical position `z` from ROMS vertical coordinates.

    Args:
        z: Binding for the particle property to store the computed physical vertical position.
        hc: Binding for the critical depth scalar
        NZ: Binding for the scalar giving the number of vertical rho levels.
        h: Binding for the bathymetry field
        zeta: Binding for the sea surface height field
        C: Binding for the vertical stretching function field
    """
    kernel = ParticleKernel(
        compute_z_kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ZIDX_DECLARATION,
            YIDX_DECLARATION,
            XIDX_DECLARATION,
            z_declaration,
        ],
        scalars=[
            hc_declaration,
            NZ_declaration,
        ],
        field_data=[
            h_declaration,
            zeta_declaration,
            C_declaration,
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "z": z,
        },
        scalar_bindings={
            "hc": hc,
            "NZ": NZ,
        },
        field_data_bindings={
            "h": h,
            "zeta": zeta,
            "C": C,
        },
    )


def construct_compute_zidx_kernel(
    z: str = "z",
    hc: str = "hc",
    NZ: str = "NZ",
    h: str = "h",
    zeta: str = "zeta",
    C: str = "C",
) -> BoundKernel:
    """Construct a kernel to compute the ROMS vertical index `zidx` from physical vertical position `z`.

    Args:
        z: Binding for the particle property giving the physical vertical position.
        hc: Binding for the critical depth scalar
        NZ: Binding for the scalar giving the number of vertical rho levels.
        h: Binding for the bathymetry field
        zeta: Binding for the sea surface height field
        C: Binding for the vertical stretching function field
    """
    kernel = ParticleKernel(
        compute_zidx_kernel_function,
        particle_properties=[STATUS_DECLARATION, ZIDX_DECLARATION, YIDX_DECLARATION, XIDX_DECLARATION, z_declaration],
        scalars=[
            hc_declaration,
            NZ_declaration,
        ],
        field_data=[
            h_declaration,
            zeta_declaration,
            C_declaration,
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "z": z,
        },
        scalar_bindings={
            "hc": hc,
            "NZ": NZ,
        },
        field_data_bindings={
            "h": h,
            "zeta": zeta,
            "C": C,
        },
    )
