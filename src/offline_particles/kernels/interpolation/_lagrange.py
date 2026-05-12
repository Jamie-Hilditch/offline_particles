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
def _lagrange_basis_polynomial(N: int) -> Callable[[float, int], float]:
    """Return a function that computes the j-th Lagrange basis polynomial of degree 2N-1 at x."""

    @numba.njit(nogil=True, fastmath=True)
    def impl(x: float, j: int) -> float:
        lbp = 1.0
        for k in range(2 * N):
            if j == k:
                continue
            lbp *= (x - k) / (j - k)
        return lbp

    return impl


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

    Warning
    -------
    This function is intended to be used in hot loops and thus does not check if max_idx is compatible with the field array size.
    I.e. 0 <= max_idx <= field.shape[0] - 1 must hold to avoid out-of-bounds memory access.
    It is the caller's responsibility to ensure that the field array has at least 2N points to avoid out-of-bounds memory access.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    lagrange_2N = _lagrange_basis_polynomial(N)

    @numba.njit(nogil=True, fastmath=True)
    def impl(
        field_array: npt.NDArray[np.generic],
        offset_idx: np.float64,
        max_idx: int,  # max index for the lower index to avoid out-of-bounds
    ) -> np.generic:
        """Implementation of a 2N point Lagrange interpolating polynomial in 1D for a single particle."""
        # work in the field array's dtype to avoid unnecessary casts and preserve precision
        scalar_t = field_array.dtype.type

        # get integer and fractional parts of the index
        shifted_idx = offset_idx - N
        I0 = _truncate_index(shifted_idx, max_idx)
        x0 = N + shifted_idx - I0

        # accumulate value
        value = scalar_t(0.0)
        for i in range(2 * N):
            l0 = scalar_t(lagrange_2N(x0, i))
            value += l0 * field_array[I0 + i]
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
        field_array: npt.NDArray[np.generic],
        offset: float,
    ) -> None:
        """Implementation of a 2N point Lagrange interpolating polynomial in 1D."""
        max_idx = field_array.shape[0] - 2 * N  # max index for the lower index to avoid out-of-bounds
        if max_idx < 0:
            raise ValueError(
                "Field array must have at least 2N points in the relevant dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            offset_idx = idx[i] + offset

            # accumulate is a compile-time constant so the dead branch is eliminated
            if accumulate:
                output[i] += single_particle_interpolator(field_array, offset_idx, max_idx)
            else:
                output[i] = single_particle_interpolator(field_array, offset_idx, max_idx)

    return impl


@functools.lru_cache(maxsize=None)
def lagrange2N_2D_particle_factory(N: int) -> Callable[[np.float64, np.float64, npt.NDArray[np.generic]], np.generic]:
    """Factory function for 2D Lagrange polynomial interpolation of a single particle on a 2N point stencil.

    Parameters
    ----------
    N : int
        The number of points on either side of the lower index to use for interpolation. The total number of points used for interpolation will be 2N.

    Returns
    -------
    Callable
        A JIT compiled function implementing 2D Lagrange polynomial interpolation for a single particle.

    Notes
    -----
    This factory function is cached such that the same function will be returned for the same value of N. This saves recompiling the function if reused.

    Warning
    -------
    This function is intended to be used in hot loops and thus does not check if max_idx_{i} is compatible with the field array size.
    I.e. 0 <= max_idx_{i} <= field.shape[i] - 1 must hold to avoid out-of-bounds memory access.
    It is the caller's responsibility to ensure that the field array has at least 2N points in each dimension to avoid out-of-bounds memory access.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    stencil_size = 2 * N
    lagrange_2N = _lagrange_basis_polynomial(N)

    @numba.njit(nogil=True, fastmath=True)
    def impl(
        field_array: npt.NDArray[np.generic],
        offset_idx_0: np.float64,
        offset_idx_1: np.float64,
        max_idx_0: int,
        max_idx_1: int,
    ) -> np.generic:
        """Implementation of a 2N point Lagrange interpolating polynomial in 2D for a single particle."""
        # work in the field array's dtype to avoid unnecessary casts and preserve precision
        scalar_t = field_array.dtype.type

        # get integer and fractional parts of the index
        shifted_idx_0 = offset_idx_0 - N
        shifted_idx_1 = offset_idx_1 - N
        I0 = _truncate_index(shifted_idx_0, max_idx_0)
        I1 = _truncate_index(shifted_idx_1, max_idx_1)
        x0 = N + shifted_idx_0 - I0
        x1 = N + shifted_idx_1 - I1

        # compute lagrange basis polynomial weights
        # the branching is ugly but N is a compile-time constant and with this numba can use registers for N = 1, 2 and 3
        if N == 1:
            l0 = (scalar_t(lagrange_2N(x0, 0)), scalar_t(lagrange_2N(x0, 1)))
            l1 = (scalar_t(lagrange_2N(x1, 0)), scalar_t(lagrange_2N(x1, 1)))
        elif N == 2:
            l0 = (
                scalar_t(lagrange_2N(x0, 0)),
                scalar_t(lagrange_2N(x0, 1)),
                scalar_t(lagrange_2N(x0, 2)),
                scalar_t(lagrange_2N(x0, 3)),
            )
            l1 = (
                scalar_t(lagrange_2N(x1, 0)),
                scalar_t(lagrange_2N(x1, 1)),
                scalar_t(lagrange_2N(x1, 2)),
                scalar_t(lagrange_2N(x1, 3)),
            )
        elif N == 3:
            l0 = (
                scalar_t(lagrange_2N(x0, 0)),
                scalar_t(lagrange_2N(x0, 1)),
                scalar_t(lagrange_2N(x0, 2)),
                scalar_t(lagrange_2N(x0, 3)),
                scalar_t(lagrange_2N(x0, 4)),
                scalar_t(lagrange_2N(x0, 5)),
            )
            l1 = (
                scalar_t(lagrange_2N(x1, 0)),
                scalar_t(lagrange_2N(x1, 1)),
                scalar_t(lagrange_2N(x1, 2)),
                scalar_t(lagrange_2N(x1, 3)),
                scalar_t(lagrange_2N(x1, 4)),
                scalar_t(lagrange_2N(x1, 5)),
            )
        else:
            l0 = np.empty(stencil_size, dtype=scalar_t)
            l1 = np.empty(stencil_size, dtype=scalar_t)
            for i in range(stencil_size):
                l0[i] = lagrange_2N(x0, i)
                l1[i] = lagrange_2N(x1, i)

        # accumulate value
        value = scalar_t(0.0)
        for i in range(stencil_size):
            for j in range(stencil_size):
                value += l0[i] * l1[j] * field_array[I0 + i, I1 + j]
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
        field_array: npt.NDArray[np.generic],
        offset0: float,
        offset1: float,
    ) -> None:
        """Implementation of a 2N point Lagrange interpolating polynomial in 2D."""
        max_idx_0 = field_array.shape[0] - 2 * N  # max index for the lower index to avoid out-of-bounds
        max_idx_1 = field_array.shape[1] - 2 * N  # max index for the lower index to avoid out-of-bounds
        if max_idx_0 < 0 or max_idx_1 < 0:
            raise ValueError(
                "Field array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
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

    return impl


@functools.lru_cache(maxsize=None)
def lagrange2N_3D_particle_factory(
    N: int,
) -> Callable[[np.float64, np.float64, np.float64, npt.NDArray[np.generic]], np.generic]:
    """Factory function for 3D Lagrange polynomial interpolation of a single particle on a 2N point stencil.

    Parameters
    ----------
    N : int
        The number of points on either side of the lower index to use for interpolation. The total number of points used for interpolation will be 2N.

    Returns
    -------
    Callable
        A JIT compiled function implementing 3D Lagrange polynomial interpolation for a single particle.

    Notes
    -----
    This factory function is cached such that the same function will be returned for the same value of N. This saves recompiling the function if reused.

    Warning
    -------
    This function is intended to be used in hot loops and thus does not check if max_idx_{i} is compatible with the field array size.
    I.e. 0 <= max_idx_{i} <= field.shape[i] - 1 must hold to avoid out-of-bounds memory access.
    It is the caller's responsibility to ensure that the field array has at least 2N points in each dimension to avoid out-of-bounds memory access.
    """

    if N <= 0:
        raise ValueError("N must be a positive integer.")

    stencil_size = 2 * N
    lagrange_2N = _lagrange_basis_polynomial(N)

    @numba.njit(nogil=True, fastmath=True)
    def impl(
        field_array: npt.NDArray[np.generic],
        offset_idx_0: np.float64,
        offset_idx_1: np.float64,
        offset_idx_2: np.float64,
        max_idx_0: int,
        max_idx_1: int,
        max_idx_2: int,
    ) -> np.generic:
        """Implementation of a 2N point Lagrange interpolating polynomial in 3D for a single particle."""
        # work in the field array's dtype to avoid unnecessary casts and preserve precision
        scalar_t = field_array.dtype.type

        # get integer and fractional parts of the index
        shifted_idx_0 = offset_idx_0 - N
        shifted_idx_1 = offset_idx_1 - N
        shifted_idx_2 = offset_idx_2 - N
        I0 = _truncate_index(shifted_idx_0, max_idx_0)
        I1 = _truncate_index(shifted_idx_1, max_idx_1)
        I2 = _truncate_index(shifted_idx_2, max_idx_2)
        x0 = N + shifted_idx_0 - I0
        x1 = N + shifted_idx_1 - I1
        x2 = N + shifted_idx_2 - I2

        # compute lagrange basis polynomial weights
        # the branching is ugly but N is a compile-time constant and with this numba can use registers for N = 1, 2 and 3
        if N == 1:
            l0 = (scalar_t(lagrange_2N(x0, 0)), scalar_t(lagrange_2N(x0, 1)))
            l1 = (scalar_t(lagrange_2N(x1, 0)), scalar_t(lagrange_2N(x1, 1)))
            l2 = (scalar_t(lagrange_2N(x2, 0)), scalar_t(lagrange_2N(x2, 1)))
        elif N == 2:
            l0 = (
                scalar_t(lagrange_2N(x0, 0)),
                scalar_t(lagrange_2N(x0, 1)),
                scalar_t(lagrange_2N(x0, 2)),
                scalar_t(lagrange_2N(x0, 3)),
            )
            l1 = (
                scalar_t(lagrange_2N(x1, 0)),
                scalar_t(lagrange_2N(x1, 1)),
                scalar_t(lagrange_2N(x1, 2)),
                scalar_t(lagrange_2N(x1, 3)),
            )
            l2 = (
                scalar_t(lagrange_2N(x2, 0)),
                scalar_t(lagrange_2N(x2, 1)),
                scalar_t(lagrange_2N(x2, 2)),
                scalar_t(lagrange_2N(x2, 3)),
            )
        elif N == 3:
            l0 = (
                scalar_t(lagrange_2N(x0, 0)),
                scalar_t(lagrange_2N(x0, 1)),
                scalar_t(lagrange_2N(x0, 2)),
                scalar_t(lagrange_2N(x0, 3)),
                scalar_t(lagrange_2N(x0, 4)),
                scalar_t(lagrange_2N(x0, 5)),
            )
            l1 = (
                scalar_t(lagrange_2N(x1, 0)),
                scalar_t(lagrange_2N(x1, 1)),
                scalar_t(lagrange_2N(x1, 2)),
                scalar_t(lagrange_2N(x1, 3)),
                scalar_t(lagrange_2N(x1, 4)),
                scalar_t(lagrange_2N(x1, 5)),
            )
            l2 = (
                scalar_t(lagrange_2N(x2, 0)),
                scalar_t(lagrange_2N(x2, 1)),
                scalar_t(lagrange_2N(x2, 2)),
                scalar_t(lagrange_2N(x2, 3)),
                scalar_t(lagrange_2N(x2, 4)),
                scalar_t(lagrange_2N(x2, 5)),
            )
        else:
            l0 = np.empty(stencil_size, dtype=scalar_t)
            l1 = np.empty(stencil_size, dtype=scalar_t)
            l2 = np.empty(stencil_size, dtype=scalar_t)
            for i in range(stencil_size):
                l0[i] = lagrange_2N(x0, i)
                l1[i] = lagrange_2N(x1, i)
                l2[i] = lagrange_2N(x2, i)

        # accumulate value
        value = scalar_t(0.0)
        for i in range(2 * N):
            for j in range(2 * N):
                for k in range(2 * N):
                    value += l0[i] * l1[j] * l2[k] * field_array[I0 + i, I1 + j, I2 + k]
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
        field_array: npt.NDArray[np.generic],
        offset0: float,
        offset1: float,
        offset2: float,
    ) -> None:
        """Implementation of a 2N point Lagrange interpolating polynomial in 3D."""
        max_idx_0 = field_array.shape[0] - 2 * N  # max index for the lower index to avoid out-of-bounds
        max_idx_1 = field_array.shape[1] - 2 * N  # max index for the lower index to avoid out-of-bounds
        max_idx_2 = field_array.shape[2] - 2 * N  # max index for the lower index to avoid out-of-bounds
        if max_idx_0 < 0 or max_idx_1 < 0 or max_idx_2 < 0:
            raise ValueError(
                "Field array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
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

    return impl
