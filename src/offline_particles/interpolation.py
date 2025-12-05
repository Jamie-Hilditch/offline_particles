"""Submodule defining interpolation routines for field data."""

import numba
import numpy.typing as npt

from .kernel_tools import split_index


@numba.njit(nogil=True, fastmath=True)
def linear_interpolation(idx: tuple[float], array: npt.NDArray[float]) -> float:
    """Perform linear interpolation on a 1D array.
    
    Parameters:
        idx: The floating-point index to interpolate at.
        array: 1D array of values to interpolate within.
    
    Returns:
        The interpolated value as a float.
    """
    I0, f0 = split_index(idx[0])
    g0 = 1.0 - f0

    v0 = array[I0]
    v1 = array[I0 + 1]

    return g0 * v0 + f0 * v1

@numba.njit(nogil=True, fastmath=True)
def bilinear_interpolation(idx: tuple[float, float], array: npt.NDArray[float]) -> float:
    """Perform bilinear interpolation on a 2D array.
    
    Parameters:
        idx: The floating-point indices.
        array: 2D array of values to interpolate within.
    
    Returns:
        The interpolated value as a float.
    """
    I0, f0 = split_index(idx[0])
    I1, f1 = split_index(idx[1])
    g0 = 1.0 - f0
    g1 = 1.0 - f1

    v00 = array[I0, I1]
    v01 = array[I0, I1 + 1]
    v10 = array[I0 + 1, I1]
    v11 = array[I0 + 1, I1 + 1]

    return (
        g0 * g1 * v00 +
        g0 * f1 * v01 +
        f0 * g1 * v10 +
        f0 * f1 * v11
    )

@numba.njit(nogil=True, fastmath=True)
def trilinear_interpolation(
    idx: tuple[float, float, float], array: npt.NDArray[float]
) -> float:
    """Perform trilinear interpolation on a 3D array.
    
    Parameters:
        idx: The floating-point indices.
        array: 3D array of values to interpolate within.
    
    Returns:
        The interpolated value as a float.
    """
    I0, f0 = split_index(idx[0])
    I1, f1 = split_index(idx[1])
    I2, f2 = split_index(idx[2])
    g0 = 1.0 - f0
    g1 = 1.0 - f1
    g2 = 1.0 - f2

    v000 = array[I0, I1, I2]
    v001 = array[I0, I1, I2 + 1]
    v010 = array[I0, I1 + 1, I2]
    v011 = array[I0, I1 + 1, I2 + 1]
    v100 = array[I0 + 1, I1, I2]
    v101 = array[I0 + 1, I1, I2 + 1]
    v110 = array[I0 + 1, I1 + 1, I2]
    v111 = array[I0 + 1, I1 + 1, I2 + 1]

    return (
        g0 * g1 * g2 * v000 +
        g0 * g1 * f2 * v001 +
        g0 * f1 * g2 * v010 +
        g0 * f1 * f2 * v011 +
        f0 * g1 * g2 * v100 +
        f0 * g1 * f2 * v101 +
        f0 * f1 * g2 * v110 +
        f0 * f1 * f2 * v111
    )