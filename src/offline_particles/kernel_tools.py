"""A submodule with utility functions for kernel operations."""

import numba
import numpy as np
import numpy.typing as npt


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
def offset_indices_1D(pidx0: float, offsets: tuple[float]) -> float:
    """Offset a particle index."""
    return (pidx0 + offsets[0],)


@numba.njit(nogil=True, fastmath=True)
def offset_indices_2D(
    pidx0: float, pidx1: float, offsets: tuple[float, float]
) -> tuple[float, float]:
    """Offset  2 particle indices."""
    return pidx0 + offsets[0], pidx1 + offsets[1]


@numba.njit(nogil=True, fastmath=True)
def offset_indices_3D(
    pidx0: float, pidx1: float, pidx2: float, offsets: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Offset 3 particle indices."""
    return pidx0 + offsets[0], pidx1 + offsets[1], pidx2 + offsets[2]


@numba.njit(nogil=True, fastmath=True)
def split_index(idx: float, max_idx: int) -> tuple[int, float]:
    """Split a single index into its integer and fractional parts.

    Parameters:
        idx: The index to split.
    Returns:
        A tuple containing the integer part and the fractional part.

    """
    int_idx = int(np.floor(idx))
    int_idx = max(0, int_idx)
    int_idx = min(max_idx, int_idx)
    frac_idx = idx - int_idx
    return int_idx, frac_idx
