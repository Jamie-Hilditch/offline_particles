"""Submodule for handling vertical coordinate transformations in ROMS."""

import numba
import numpy as np
import numpy.typing as npt

from ..kernel_tools import unsafe_inverse_linear_interpolation


@numba.njit(nogil=True, fastmath=True)
def sigma_coordinate(zidx: float, Nz: int) -> float:
    """Compute the sigma coordinate from a vertical index.

    Parameters:
        zidx: The vertical index (float)
        Nz: Total number of vertical levels (int)
    Returns:
        The sigma coordinate (float)
    """
    return (zidx + 0.5) / Nz - 1.0

@numba.njit(nogil=True, fastmath=True)
def S_coordinate(hc: float, sigma: float, h: float, C: float) -> float:
    """Compute the S-coordinate transformation for ROMS vertical coordinates.

    Parameters:
        hc: Critical depth (float)
        sigma: Sigma coordinate (float)
        h: Bathymetric depth (float)
        C: Stretching function value (float)
    Returns:
        The S-coordinate value (float)
    """
    return (hc * sigma + h * C) / (hc + h)


@numba.njit(nogil=True, fastmath=True)
def z_from_S(S: float, h: float, zeta: float) -> float:
    """Convert S-coordinate to physical coordinate.

    Parameters:
        S: S-coordinate value (float)
        h: Bathymetric depth (float)
        zeta: Free surface elevation (float)

    Returns:
        z (float)
    """
    return zeta + (zeta + h) * S


@numba.njit(nogil=True, fastmath=True)
def S_from_z(z: float, h: float, zeta: float) -> float:
    """Convert physical coordinate to S-coordinate.

    Parameters:
        z: Physical coordinate (float)
        h: Bathymetric depth (float)
        zeta: Free surface elevation (float)

    Returns:
        S-coordinate value (float)
    """
    return (z - zeta) / (zeta + h)

@numba.njit(nogil=True, fastmath=True)
def compute_zidx_from_S(
    S: float,
    hc: float,
    NZ: int,
    h_value: float,
    zeta_value: float,
    C: npt.NDArray[float],
    C_offset: npt.NDArray[float],
) -> float:
    """Compute zidx from S-coordinate."""
    C_size = C.shape[0]

    # compute sigma and S arrays
    rho_indices = np.arange(C_size, dtype=np.float64) - C_offset
    sigma_array = (rho_indices + 0.5) / NZ - 1.0
    S_array = S_coordinate(hc, sigma_array, h_value, C)

    # inverse interpolation to get zidx
    if S <= S_array[0]:
        S0 = S_array[0]
        S1 = S_array[1]
        Sidx = (S - S0) / (S1 - S0)
    elif S >= S_array[C_size - 1]:
        S0 = S_array[C_size - 2]
        S1 = S_array[C_size - 1]
        Sidx = C_size - 2 + (S - S0) / (S1 - S0)
    else:
        Sidx = unsafe_inverse_linear_interpolation(S_array, S)
    zidx = Sidx + C_offset
    
    return zidx