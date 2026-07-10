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
from ._vertical_coordinate import compute_z_kernel_function, compute_zidx_kernel_function

__all__ = [
    "construct_compute_z_kernel",
    "construct_compute_zidx_kernel",
]

z_declaration = ParticlePropertyDeclaration("z", np.floating)
hc_declaration = ScalarDeclaration("hc", np.floating)
NZ_declaration = ScalarDeclaration("NZ", np.integer)
h_declaration = FieldDataDeclaration("h", np.floating, [validate_YX_ordering])
zeta_declaration = FieldDataDeclaration("zeta", np.floating, [validate_YX_ordering])
C_declaration = FieldDataDeclaration("C", np.floating, [validate_Z_ordering])


def construct_compute_z_kernel(
    z: str = "z",
    hc: str = "hc",
    NZ: str = "NZ",
    h: str = "h",
    zeta: str = "zeta",
    C: str = "C",
) -> BoundKernel:
    """Construct a kernel to compute the physical vertical position `z` from ROMS vertical coordinates.

    Parameters
    ----------
    z : str, optional
        Binding for the particle property to store the computed physical vertical position.
    hc : str, optional
        Binding for the critical depth scalar
    NZ : str, optional
        Binding for the scalar giving the number of vertical rho levels.
    h : str, optional
        Binding for the bathymetry field
    zeta : str, optional
        Binding for the sea surface height field
    C : str, optional
        Binding for the vertical stretching function field. Must be strictly increasing.

    Returns
    -------
    BoundKernel
        A bound kernel that computes the physical vertical position `z`.
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

    Parameters
    ----------
    z : str, optional
        Binding for the particle property giving the physical vertical position.
    hc : str, optional
        Binding for the critical depth scalar
    NZ : str, optional
        Binding for the scalar giving the number of vertical rho levels.
    h : str, optional
        Binding for the bathymetry field
    zeta : str, optional
        Binding for the sea surface height field
    C : str, optional
        Binding for the vertical stretching function field. Must be strictly increasing; a
        binary search over `C` is used to invert the S-coordinate transform, and behavior is
        undefined if this precondition does not hold. This is a physical requirement of the ROMS
        S-coordinate system (a valid vertical grid never has degenerate or crossing levels), so
        it should always be satisfied by real ROMS output.

    Returns
    -------
    BoundKernel
        A bound kernel that computes the ROMS vertical index `zidx` from the physical vertical position `z`.
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
