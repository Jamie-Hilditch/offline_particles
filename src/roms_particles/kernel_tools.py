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
