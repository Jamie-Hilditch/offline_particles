"""Implementations of Lagrange polynomial interpolation that map particle indices.

These functions accept all three particle indices in the order (zidx, yidx, xidx) and map them
to the correct order for interpolation based on the field array layout.
This allows us to use the same interpolation functions for different field array layouts
without needing to write separate interpolation functions for each layout.
"""

from collections.abc import Callable

import numba
import numpy as np
import numpy.typing as npt

from ...spatial_arrays import ArrayAxis
from ._lagrange import lagrange2N_1D_particle_factory, lagrange2N_2D_particle_factory, lagrange2N_3D_particle_factory

_AXIS_ENUMERATION = {ArrayAxis.Z: 0, ArrayAxis.Y: 1, ArrayAxis.X: 2}


def _lagrange2N_1D_mapped_particle_factory(axis_index: int, N: int) -> Callable:
    """Create a 1D field interpolator for the given axis and interpolation stencil size.

    Parameters
    ----------
    axis_index : int
        The index of the axis in (zidx, yidx, xidx).
    N : int
        The half-width of the interpolation stencil.

    Returns
    -------
    Callable
        A 1D field interpolator.
    """
    interpolator = lagrange2N_1D_particle_factory(N)

    @numba.njit(nogil=True, fastmath=True)
    def lagrange2N_1D_mapped_particle(
        zidx: np.float64,
        yidx: np.float64,
        xidx: np.float64,
        field_array: npt.NDArray[np.inexact],
        offset: float,
        max_idx: int,
    ):
        idx = (zidx, yidx, xidx)[axis_index]
        return interpolator(field_array, idx + offset, max_idx)

    return lagrange2N_1D_mapped_particle


def _lagrange2N_2D_mapped_particle_factory(axis_index_0: int, axis_index_1: int, N: int) -> Callable:
    """Create a 2D field interpolator for the given axes and interpolation stencil size.

    Parameters
    ----------
    axis_index_0 : int
        The index of the first axis in (zidx, yidx, xidx).
    axis_index_1 : int
        The index of the second axis in (zidx, yidx, xidx).
    N : int
        The half-width of the interpolation stencil.


    Returns
    -------
    Callable
        A 2D field interpolator.
    """
    interpolator = lagrange2N_2D_particle_factory(N)

    @numba.njit(nogil=True, fastmath=True)
    def lagrange2N_2D_mapped_particle(
        zidx: np.float64,
        yidx: np.float64,
        xidx: np.float64,
        field_array: npt.NDArray[np.inexact],
        offset_0: float,
        offset_1: float,
        max_idx_0: int,
        max_idx_1: int,
    ):
        idx0 = (zidx, yidx, xidx)[axis_index_0]
        idx1 = (zidx, yidx, xidx)[axis_index_1]
        return interpolator(field_array, idx0 + offset_0, idx1 + offset_1, max_idx_0, max_idx_1)

    return lagrange2N_2D_mapped_particle


def _lagrange2N_3D_mapped_particle_factory(axis_index_0: int, axis_index_1: int, axis_index_2: int, N: int) -> Callable:
    """Create a 3D field interpolator for the given interpolation stencil size.

    Parameters
    ----------
    axis_index_0 : int
        The index of the first axis in (zidx, yidx, xidx).
    axis_index_1 : int
        The index of the second axis in (zidx, yidx, xidx).
    axis_index_2 : int
        The index of the third axis in (zidx, yidx, xidx).
    N : int
        The half-width of the interpolation stencil.

    Returns
    -------
    Callable
        A 3D field interpolator.
    """
    interpolator = lagrange2N_3D_particle_factory(N)

    @numba.njit(nogil=True, fastmath=True)
    def lagrange2N_3D_mapped_particle(
        zidx: np.float64,
        yidx: np.float64,
        xidx: np.float64,
        field_array: npt.NDArray[np.inexact],
        offset_0: float,
        offset_1: float,
        offset_2: float,
        max_idx_0: int,
        max_idx_1: int,
        max_idx_2: int,
    ):
        idx0 = (zidx, yidx, xidx)[axis_index_0]
        idx1 = (zidx, yidx, xidx)[axis_index_1]
        idx2 = (zidx, yidx, xidx)[axis_index_2]
        return interpolator(
            field_array, idx0 + offset_0, idx1 + offset_1, idx2 + offset_2, max_idx_0, max_idx_1, max_idx_2
        )

    return lagrange2N_3D_mapped_particle


def lagrange2N_mapped_particle_factory(dim_ordering: tuple[ArrayAxis, ...], N: int) -> Callable:
    """Create a field interpolator for the given field array layout and interpolation stencil size.

    Parameters
    ----------
    dim_ordering : tuple[ArrayAxis, ...]
        The ordering of dimensions in the field array.
    N : int
        The half-width of the interpolation stencil.

    Returns
    -------
    Callable
        A field interpolator.

    Raises
    ------
    ValueError
        If there are duplicate dimensions in `dim_ordering`.
        If the number of dimensions in the field array is not supported.
    """
    # check the dim_ordering is unique
    if len(set(dim_ordering)) != len(dim_ordering):
        raise ValueError(
            f"Duplicate dimensions in dim_ordering: {dim_ordering}. Each dimension should only appear once."
        )
    # tuple of indices for the axes in the field array, numba should treat this as a compile time constant
    axis_indices = tuple(_AXIS_ENUMERATION[axis] for axis in dim_ordering)

    match len(axis_indices):
        case 1:
            return _lagrange2N_1D_mapped_particle_factory(axis_indices[0], N)
        case 2:
            return _lagrange2N_2D_mapped_particle_factory(axis_indices[0], axis_indices[1], N)
        case 3:
            return _lagrange2N_3D_mapped_particle_factory(axis_indices[0], axis_indices[1], axis_indices[2], N)
        case _:
            raise ValueError(
                f"Unsupported number of dimensions: {len(axis_indices)}. Supported number of dimensions are 1, 2, or 3."
            )
