"""A submodule with utility functions for kernel operations."""

from numbers import Number

import numba
import numpy as np
import numpy.typing as npt

from .kernel_data import KernelData


@numba.njit(nogil=True, fastmath=True)
def unsafe_inverse_linear_interpolation(array: npt.NDArray, value: float) -> float:
    """Perform an unsafe inverse linear interpolation on a 1D array.

    Parameters:
        array: 1D array of strictly increasing values to interpolate within.
        value: The value to find the corresponding index for.

    Returns:
        The interpolated index as a float.

    Note:
        This function assumes that the value is within the bounds of the array.
        No bounds checking is performed for performance reasons.
    """
    idx = np.searchsorted(array, value, side="right") - 1
    x0 = array[idx]
    x1 = array[idx + 1]
    fraction = (value - x0) / (x1 - x0)
    return idx + fraction

@numba.njit(nogil=True, fastmath=True)
def unwrap_scalar(kernel_data: KernelData) -> Number:
    """Unwrap a scalar kernel value from KernelData.

    Parameters:
        kernel_data: KernelData object containing the kernel values.

    Returns:
        The scalar kernel value.

    Raises:
        ValueError: If the kernel data does not contain exactly one value.
    """
    return kernel_data.array[()]

@numba.njit(nogil=True, fastmath=True)
def offset_indices(particle_indices: tuple[float, float, float], kernel_data: KernelData) -> npt.NDArray[float]:
    """Compute offset indices for a particle based on kernel data.
    
    Parameters:
        particle: numpy scalar containing particle data.
        kernel_data: KernelData.

    Returns:
        A tuple of numpy scalars representing the offset indices.
    """
    dmask = kernel_data.dmask
    offsets = kernel_data.offsets

    M = len(offsets)
    out = np.empty(M, dtype=np.float64)
    m = 0

    for i in range(3):
        if dmask[i] == 0:
            continue 
        out[m] = particle_indices[i] + offsets[m]
        m += 1

    return out

@numba.njit(nogil=True, fastmath=True)
def split_index(idx: float) -> tuple[int, float]:
    """Split a single index into its integer and fractional parts.

    Parameters:
        idx: The index to split.
    Returns:
        A tuple containing the integer part and the fractional part.

    """
    int_idx = np.floor(idx)
    frac_idx = idx - int_idx
    return int(int_idx), frac_idx
    