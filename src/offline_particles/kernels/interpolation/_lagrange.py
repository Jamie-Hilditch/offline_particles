"""Core functions for interpolation kernels."""

import functools
from typing import Callable

import numba
import numpy as np
import numpy.typing as npt

from ..status import INACTIVE_FLAG

__all__ = [
    "lagrange2N_1D_particle_factory",
    "lagrange2N_2D_particle_factory",
    "lagrange2N_3D_particle_factory",
    "lagrange2N_1D_factory",
    "lagrange2N_2D_factory",
    "lagrange2N_3D_factory",
]


@numba.njit(nogil=True, fastmath=True)
def _truncate_index(idx: float, max_idx: int) -> int:
    """Truncate the index to be within the bounds of the field array."""
    idx = int(idx)  # floor the index to get the lower index
    if idx < 0:
        return 0
    elif idx > max_idx:
        return max_idx
    else:
        return idx


@functools.lru_cache(maxsize=None)
def lagrange2N_1D_particle_factory(N: int) -> Callable[[np.float64, npt.NDArray[np.generic]], np.generic]:
    """Factory function for 1D Lagrange polynomial interpolation of a single particle on a 2N point stencil.

    Parameters
    ----------
    N : int
        The number of points on either side of the lower index to use for interpolation. The total number of points used for interpolation will be 2N.

    Returns
    -------
    Callable
        A JIT compiled function implementing the 1D Lagrange polynomial interpolation for a single particle.

    Notes
    -----
    This factory function is cached such that the same function will be returned for the same value of N. This saves recompiling the function if reused.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    @numba.njit(nogil=True, fastmath=True)
    def impl(
        offset_idx: np.float64,
        field: npt.NDArray[np.generic],
    ) -> np.generic:
        """Implementation of a 2N point Lagrange interpolating polynomial in 1D for a single particle."""
        # get integer and fractional parts of the index
        shifted_idx = offset_idx - N
        max_idx = field.shape[0] - 2 * N  # max index for the lower index to avoid out-of-bounds
        I0 = _truncate_index(shifted_idx, max_idx)
        x0 = N + shifted_idx - I0

        # compute the Lagrange basis polynomials
        # ∏_{j=0,1,...,2N-1 j!=k} (x0 - j) / (k - j)
        l0 = np.ones((2 * N,), dtype=np.float64)
        for k in range(
            2 * N
        ):  # np.nditer might look cleaner but Claude doesn't think numba can unroll the loops in that case
            for j in range(2 * N):
                if j == k:
                    continue
                l0[k] *= (x0 - j) / (k - j)

        value = field[0] * 0  # initialise to zero but preserve dtype of field
        # compute interpolated value
        for i in range(2 * N):
            value += l0[i] * field[I0 + i]

        return value

    return impl


@functools.lru_cache(maxsize=None)
def lagrange2N_1D_factory(
    N: int,
    accumulate: bool = False,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.generic],
        npt.NDArray[np.generic],
        float,
    ],
    None,
]:
    """Factory function creating a function implementing 1D Lagrange polynomial interpolation on a 2N point stencil.

    Parameters
    ----------
    N : int
        The number of points on either side of the lower index to use for interpolation. The total number of points used for interpolation will be 2N.
    accumulate : bool, optional
        Whether to accumulate the interpolated value to the output array instead of overwriting it. Default is False.

    Returns
    -------
    Callable
        A JIT compiled function implementing the 1D Lagrange polynomial interpolation.

    Notes
    -----
    This factory function is cached such that the same function will be returned for the same values of N and accumulate.
    This saves recompiling the function if reused.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    single_particle_interpolator = lagrange2N_1D_particle_factory(N)

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx: npt.NDArray[np.float64],
        output: npt.NDArray[np.generic],
        field: npt.NDArray[np.generic],
        offset: float,
    ) -> None:
        """Implementation of a 2N point Lagrange interpolating polynomial in 1D."""
        for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            offset_idx = idx[i] + offset

            # accumulate is a compile-time constant so the dead branch is eliminated
            if accumulate:
                output[i] += single_particle_interpolator(offset_idx, field)
            else:
                output[i] = single_particle_interpolator(offset_idx, field)

    return impl


@functools.lru_cache(maxsize=None)
def lagrange2N_2D_particle_factory(N: int) -> Callable[[np.float64, np.float64, npt.NDArray[np.generic]], np.generic]:
    """Factory function for 2D Lagrange polynomial interpolation of a single particle on a 2N point stencil."""
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    @numba.njit(nogil=True, fastmath=True)
    def impl(
        offset_idx_0: np.float64,
        offset_idx_1: np.float64,
        field: npt.NDArray[np.generic],
    ) -> np.generic:
        """Implementation of a 2N point Lagrange interpolating polynomial in 2D for a single particle."""
        # get integer and fractional parts of the index
        shifted_idx_0 = offset_idx_0 - N
        shifted_idx_1 = offset_idx_1 - N
        max_idx_0 = field.shape[0] - 2 * N  # max index for the lower index to avoid out-of-bounds
        max_idx_1 = field.shape[1] - 2 * N  # max index for the lower index to avoid out-of-bounds
        I0 = _truncate_index(shifted_idx_0, max_idx_0)
        I1 = _truncate_index(shifted_idx_1, max_idx_1)
        x0 = N + shifted_idx_0 - I0
        x1 = N + shifted_idx_1 - I1

        # compute the Lagrange basis polynomials
        # ∏_{j=0,1,...,2N-1 j!=k} (x0 - j) / (k - j)
        l0 = np.ones((2 * N,), dtype=np.float64)
        l1 = np.ones((2 * N,), dtype=np.float64)
        for k in range(
            2 * N
        ):  # np.nditer might look cleaner but Claude doesn't think numba can unroll the loops in that case
            for j in range(2 * N):
                if j == k:
                    continue
                l0[k] *= (x0 - j) / (k - j)
                l1[k] *= (x1 - j) / (k - j)

        value = field[0, 0] * 0  # initialise to zero but preserve dtype of field
        # compute interpolated value
        for i in range(2 * N):
            for j in range(2 * N):
                value += l0[i] * l1[j] * field[I0 + i, I1 + j]
        return value

    return impl


@functools.lru_cache(maxsize=None)
def lagrange2N_2D_factory(
    N: int,
    accumulate: bool = False,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.generic],
        npt.NDArray[np.generic],
        float,
        float,
    ],
    None,
]:
    """Factory function creating a function implementing 2D Lagrange polynomial interpolation on a 2N point stencil.

    Parameters
    ----------
    N : int
        The number of points on either side of the lower index to use for interpolation. The total number of points used for interpolation will be 2N.
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
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    single_particle_interpolator = lagrange2N_2D_particle_factory(N)

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        output: npt.NDArray[np.generic],
        field: npt.NDArray[np.generic],
        offset0: float,
        offset1: float,
    ) -> None:
        """Implementation of a 2N point Lagrange interpolating polynomial in 2D."""
        for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            offset_idx0 = idx0[i] + offset0
            offset_idx1 = idx1[i] + offset1

            # accumulate is a compile-time constant so the dead branch is eliminated
            if accumulate:
                output[i] += single_particle_interpolator(offset_idx0, offset_idx1, field)
            else:
                output[i] = single_particle_interpolator(offset_idx0, offset_idx1, field)

    return impl


@functools.lru_cache(maxsize=None)
def lagrange2N_3D_particle_factory(
    N: int,
) -> Callable[[np.float64, np.float64, np.float64, npt.NDArray[np.generic]], np.generic]:
    """Factory function for 3D Lagrange polynomial interpolation of a single particle on a 2N point stencil."""

    if N <= 0:
        raise ValueError("N must be a positive integer.")

    @numba.njit(nogil=True, fastmath=True)
    def impl(
        offset_idx_0: np.float64,
        offset_idx_1: np.float64,
        offset_idx_2: np.float64,
        field: npt.NDArray[np.generic],
    ) -> np.generic:
        """Implementation of a 2N point Lagrange interpolating polynomial in 3D for a single particle."""
        # get integer and fractional parts of the index
        shifted_idx_0 = offset_idx_0 - N
        shifted_idx_1 = offset_idx_1 - N
        shifted_idx_2 = offset_idx_2 - N
        max_idx_0 = field.shape[0] - 2 * N  # max index for the lower index to avoid out-of-bounds
        max_idx_1 = field.shape[1] - 2 * N  # max index for the lower index to avoid out-of-bounds
        max_idx_2 = field.shape[2] - 2 * N  # max index for the lower index to avoid out-of-bounds
        I0 = _truncate_index(shifted_idx_0, max_idx_0)
        I1 = _truncate_index(shifted_idx_1, max_idx_1)
        I2 = _truncate_index(shifted_idx_2, max_idx_2)
        x0 = N + shifted_idx_0 - I0
        x1 = N + shifted_idx_1 - I1
        x2 = N + shifted_idx_2 - I2

        l0 = np.ones((2 * N,), dtype=np.float64)
        l1 = np.ones((2 * N,), dtype=np.float64)
        l2 = np.ones((2 * N,), dtype=np.float64)
        for k in range(
            2 * N
        ):  # np.nditer might look cleaner but Claude doesn't think numba can unroll the loops in that case
            for j in range(2 * N):
                if j == k:
                    continue
                l0[k] *= (x0 - j) / (k - j)
                l1[k] *= (x1 - j) / (k - j)
                l2[k] *= (x2 - j) / (k - j)

        # compute interpolated value
        value = field[0, 0, 0] * 0  # initialise to zero but preserve dtype of field
        for i in range(2 * N):
            for j in range(2 * N):
                for k in range(2 * N):  # noqa: E741
                    value += l0[i] * l1[j] * l2[k] * field[I0 + i, I1 + j, I2 + k]
        return value

    return impl


@functools.lru_cache(maxsize=None)
def lagrange2N_3D_factory(
    N: int,
    accumulate: bool = False,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.generic],
        npt.NDArray[np.generic],
        float,
        float,
        float,
    ],
    None,
]:
    """Factory function creating a function implementing 3D Lagrange polynomial interpolation on a 2N point stencil.

    Parameters
    ----------
    N : int
        The number of points on either side of the lower index to use for interpolation. The total number of points used for interpolation will be 2N.
    accumulate : bool, optional
        Whether to accumulate the interpolated value to the output array instead of overwriting it. Default is False.

    Returns
    -------
    Callable
        A JIT compiled function implementing the 3D Lagrange polynomial interpolation.

    Notes
    -----
    This factory function is cached such that the same function will be returned for the same values of N and accumulate.
    This saves recompiling the function if reused.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    single_particle_interpolator = lagrange2N_3D_particle_factory(N)

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        idx2: npt.NDArray[np.float64],
        output: npt.NDArray[np.generic],
        field: npt.NDArray[np.generic],
        offset0: float,
        offset1: float,
        offset2: float,
    ) -> None:
        """Implementation of a 2N point Lagrange interpolating polynomial in 3D."""
        for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            offset_idx_2 = idx2[i] + offset2

            # accumulate is a compile-time constant so the dead branch is eliminated
            if accumulate:
                output[i] += single_particle_interpolator(offset_idx_0, offset_idx_1, offset_idx_2, field)
            else:
                output[i] = single_particle_interpolator(offset_idx_0, offset_idx_1, offset_idx_2, field)

    return impl
