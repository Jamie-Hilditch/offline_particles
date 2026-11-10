"""Factories for Lagrange polynomial interpolation kernel functions."""

import functools

import numba
import numpy as np
import numpy.typing as npt

from .._kernels import KernelFunction, kernel_function
from ..status import INACTIVE_FLAG
from ._lagrange_particle_implementations import (
    lagrange_1D_particle_factory,
    lagrange_2D_particle_factory,
    lagrange_3D_particle_factory,
)


@functools.cache
def _lagrange_1D_kernel_function_factory(
    N: int,
    D: int = 0,
    accumulate: bool = False,
) -> KernelFunction:
    """Create a function implementing 1D Lagrange polynomial interpolation.

    Parameters
    ----------
    N : int
        The number of points on either side of the particles to use for interpolation. The total number of points used for interpolation will be 2N.
    D : int
        The derivative order. (Default 0)
    accumulate : bool, optional
        Whether to accumulate the interpolated value to the output array instead of overwriting it. Default is False.

    Returns
    -------
    Callable
        A JIT compiled function implementing 1D Lagrange polynomial interpolation.

    Notes
    -----
    This factory function is cached such that the same function will be returned for the same values of N and accumulate.
    This saves recompiling the function if reused.
    """
    single_particle_interpolator = lagrange_1D_particle_factory(N, D)

    @kernel_function(["status", "idx", "output"], [], ["field"])
    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def lagrange_1D(
        status: npt.NDArray[np.uint8],
        idx: npt.NDArray[np.float64],
        output: npt.NDArray[np.inexact],
        field_array: npt.NDArray[np.inexact],
        offset: float,
    ) -> None:
        """Implement Lagrange polynomial interpolation in 1D.

        Parameters
        ----------
        status : npt.NDArray[np.uint8]
            The particle property array of statuses.
        idx : npt.NDArray[np.float64]
            The particle property array of indices.
        output : npt.NDArray[np.inexact]
            The particle property array to store the interpolated values.
        field_array : npt.NDArray[np.inexact]
            The 1D array of field values to interpolate from.
        offset : float
            The offset for this field.

        Raises
        ------
        ValueError
            If the field array does not have at least 2N points to avoid out-of-bounds memory access.
        """
        max_idx = field_array.shape[0] - 2 * N  # max index for the lower index to avoid out-of-bounds
        if max_idx < 0:
            raise ValueError("Field array must have at least 2N points to avoid out-of-bounds memory access.")

        for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            offset_idx = idx[i] + offset

            # accumulate is a compile-time constant so the dead branch is eliminated
            if accumulate:
                output[i] += single_particle_interpolator(field_array, offset_idx, max_idx)
            else:
                output[i] = single_particle_interpolator(field_array, offset_idx, max_idx)

    return lagrange_1D


@functools.cache
def _lagrange_2D_kernel_function_factory(
    N0: int,
    N1: int,
    D0: int = 0,
    D1: int = 0,
    accumulate: bool = False,
) -> KernelFunction:
    """Create a function implementing 2D Lagrange polynomial interpolation on a 2N point stencil.

    Parameters
    ----------
    N0 : int
        The number of points on either side of the particle to use for interpolation in dimension 0. The total number of points used for interpolation will be 2 * N0.
    N1 : int
        The number of points on either side of the particle to use for interpolation in dimension 1. The total number of points used for interpolation will be 2 * N1.
    D0 : int
        The derivative order in dimension 0. (Default 0)
    D1 : int
        The derivative order in dimension 1. (Default 0)
    accumulate : bool, optional
        Whether to accumulate the interpolated value to the output array instead of overwriting it. Default is False.

    Returns
    -------
    Callable
        A JIT compiled function implementing the 2D Lagrange polynomial interpolation.

    Notes
    -----
    This factory function is cached such that the same function will be returned for the same values of N and accumulate.
    This saves recompiling the function if reused.
    """
    single_particle_interpolator = lagrange_2D_particle_factory(N0, N1, D0, D1)

    @kernel_function(["status", "idx_0", "idx_1", "output"], [], ["field"])
    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def lagrange_2D(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        output: npt.NDArray[np.inexact],
        field_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
    ) -> None:
        """Implement a 2N point Lagrange interpolating polynomial in 2D.

        Parameters
        ----------
        status : npt.NDArray[np.uint8]
            The particle property array of statuses.
        idx0 : npt.NDArray[np.float64]
            The particle property array of indices in the first dimension.
        idx1 : npt.NDArray[np.float64]
            The particle property array of indices in the second dimension.
        output : npt.NDArray[np.inexact]
            The particle property array to store the interpolated values.
        field_array : npt.NDArray[np.inexact]
            The 2D array of field values to interpolate from.
        offset0 : float
            The offset for the first dimension.
        offset1 : float
            The offset for the second dimension.

        Raises
        ------
        ValueError
            If the field array has in sufficient points in each dimension to avoid out-of-bounds memory access.
        """
        max_idx_0 = field_array.shape[0] - 2 * N0  # max index for the lower index to avoid out-of-bounds
        max_idx_1 = field_array.shape[1] - 2 * N1  # max index for the lower index to avoid out-of-bounds
        if max_idx_0 < 0 or max_idx_1 < 0:
            raise ValueError(
                "Field array must have at least (2 N0) x (2 N1) points to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            offset_idx0 = idx0[i] + offset0
            offset_idx1 = idx1[i] + offset1

            # accumulate is a compile-time constant so the dead branch is eliminated
            if accumulate:
                output[i] += single_particle_interpolator(field_array, offset_idx0, offset_idx1, max_idx_0, max_idx_1)
            else:
                output[i] = single_particle_interpolator(field_array, offset_idx0, offset_idx1, max_idx_0, max_idx_1)

    return lagrange_2D


@functools.cache
def _lagrange_3D_kernel_function_factory(
    N0: int,
    N1: int,
    N2: int,
    D0: int = 0,
    D1: int = 0,
    D2: int = 0,
    accumulate: bool = False,
) -> KernelFunction:
    """Create a function implementing 3D Lagrange polynomial interpolation on a 2N point stencil.

    Parameters
    ----------
    N0 : int
        The number of points on either side of the particle to use for interpolation in dimension 0. The total number of points used for interpolation will be 2 * N0.
    N1 : int
        The number of points on either side of the particle to use for interpolation in dimension 1. The total number of points used for interpolation will be 2 * N1.
    N2: int
        The number of points on either side of the particle to use for interpolation in dimension 2. The total number of points used for interpolation will be 2 * N2.
    D0 : int
        The derivative order in dimension 0.
    D1 : int
        The derivative order in dimension 1.
    D2: int
        The derivative order in dimension 2.
    accumulate : bool, optional
        Whether to accumulate the interpolated value to the output array instead of overwriting it. Default is False.

    Returns
    -------
    Callable
        A JIT compiled function implementing 3D Lagrange polynomial interpolation.

    Notes
    -----
    This factory function is cached such that the same function will be returned for the same values of N and accumulate.
    This saves recompiling the function if reused.
    """
    single_particle_interpolator = lagrange_3D_particle_factory(N0, N1, N2, D0, D1, D2)

    @kernel_function(["status", "idx_0", "idx_1", "idx_2", "output"], [], ["field"])
    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def lagrange_3D(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        idx2: npt.NDArray[np.float64],
        output: npt.NDArray[np.inexact],
        field_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
        offset2: float,
    ) -> None:
        """Implement 2N point Lagrange interpolating polynomial in 3D.

        Parameters
        ----------
        status : npt.NDArray[np.uint8]
            The particle property array of statuses.
        idx0 : npt.NDArray[np.float64]
            The particle property array of indices in the first dimension.
        idx1 : npt.NDArray[np.float64]
            The particle property array of indices in the second dimension.
        idx2 : npt.NDArray[np.float64]
            The particle property array of indices in the third dimension.
        output : npt.NDArray[np.inexact]
            The particle property array to store the interpolated values.
        field_array : npt.NDArray[np.inexact]
            The 3D array of field values to interpolate from.
        offset0 : float
            The offset for the first dimension.
        offset1 : float
            The offset for the second dimension.
        offset2 : float
            The offset for the third dimension.

        Raises
        ------
        ValueError
            If the field array does not have a sufficient number of points in any dimension.
        """
        max_idx_0 = field_array.shape[0] - 2 * N0  # max index for the lower index to avoid out-of-bounds
        max_idx_1 = field_array.shape[1] - 2 * N1  # max index for the lower index to avoid out-of-bounds
        max_idx_2 = field_array.shape[2] - 2 * N2  # max index for the lower index to avoid out-of-bounds
        if max_idx_0 < 0 or max_idx_1 < 0 or max_idx_2 < 0:
            raise ValueError(
                "Field array must have at least (2 N0) x (2 N1) x (2 N2) points to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            offset_idx_2 = idx2[i] + offset2

            # accumulate is a compile-time constant so the dead branch is eliminated
            if accumulate:
                output[i] += single_particle_interpolator(
                    field_array, offset_idx_0, offset_idx_1, offset_idx_2, max_idx_0, max_idx_1, max_idx_2
                )
            else:
                output[i] = single_particle_interpolator(
                    field_array, offset_idx_0, offset_idx_1, offset_idx_2, max_idx_0, max_idx_1, max_idx_2
                )

    return lagrange_3D
