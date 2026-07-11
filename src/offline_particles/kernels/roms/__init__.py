"""Kernels for ROMS simulations."""

import numpy as np

from .._kernels import (
    BoundKernel,
    FieldDataDeclaration,
    ParticleKernel,
    ParticlePropertyDeclaration,
)
from ..input_declarations import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from ..layout_validators import validate_YX_ordering, validate_Z_ordering
from ._vertical_coordinate import compute_z_kernel_function_factory, compute_zidx_kernel_function_factory

__all__ = [
    "construct_compute_z_kernel",
    "construct_compute_zidx_kernel",
]

z_declaration = ParticlePropertyDeclaration("z", np.floating)
h_declaration = FieldDataDeclaration("h", np.floating, [validate_YX_ordering])
zeta_declaration = FieldDataDeclaration("zeta", np.floating, [validate_YX_ordering])
C_declaration = FieldDataDeclaration("C", np.floating, [validate_Z_ordering])


def construct_compute_z_kernel(
    hc: float,
    NZ: int,
    z: str = "z",
    h: str = "h",
    zeta: str = "zeta",
    C: str = "C",
) -> BoundKernel:
    """Construct a kernel to compute the physical vertical position `z` from ROMS vertical coordinates.

    Parameters
    ----------
    hc : float
        The critical depth. Baked into the compiled kernel as a compile-time constant.
    NZ : int
        The total number of vertical rho levels. Baked into the compiled kernel as a compile-time
        constant.
    z : str, optional
        Binding for the particle property to store the computed physical vertical position.
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
    kernel_fn = compute_z_kernel_function_factory(hc, NZ)
    kernel = ParticleKernel(
        kernel_fn,
        particle_properties=[
            STATUS_DECLARATION,
            ZIDX_DECLARATION,
            YIDX_DECLARATION,
            XIDX_DECLARATION,
            z_declaration,
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
        field_data_bindings={
            "h": h,
            "zeta": zeta,
            "C": C,
        },
    )


def construct_compute_zidx_kernel(
    hc: float,
    NZ: int,
    z: str = "z",
    h: str = "h",
    zeta: str = "zeta",
    C: str = "C",
) -> BoundKernel:
    """Construct a kernel to compute the ROMS vertical index `zidx` from physical vertical position `z`.

    Parameters
    ----------
    hc : float
        The critical depth. Baked into the compiled kernel as a compile-time constant.
    NZ : int
        The total number of vertical rho levels. Baked into the compiled kernel as a compile-time
        constant.
    z : str, optional
        Binding for the particle property giving the physical vertical position.
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
    kernel_fn = compute_zidx_kernel_function_factory(hc, NZ)
    kernel = ParticleKernel(
        kernel_fn,
        particle_properties=[STATUS_DECLARATION, ZIDX_DECLARATION, YIDX_DECLARATION, XIDX_DECLARATION, z_declaration],
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
        field_data_bindings={
            "h": h,
            "zeta": zeta,
            "C": C,
        },
    )
