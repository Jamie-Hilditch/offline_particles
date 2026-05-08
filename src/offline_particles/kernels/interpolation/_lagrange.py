"""Core functions for interpolation kernels."""

import functools
from typing import Callable

import numba
import numpy as np
import numpy.typing as npt

from ..status import INACTIVE_FLAG

__all__ = [
    "lagrange2N_1D_factory",
    "lagrange2N_2D_factory",
    "lagrange2N_3D_factory",
    "linear_interpolation_factory",
    "bilinear_interpolation_factory",
    "trilinear_interpolation_factory",
    "cubic_interpolation_factory",
    "bicubic_interpolation_factory",
    "tricubic_interpolation_factory",
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
    """Factory function creating a function implementing 1D Lagrange polynomial interpolation on a 2N point stencil."""

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx: npt.NDArray[np.float64],
        output: npt.NDArray[np.generic],
        field: npt.NDArray[np.generic],
        offset: float,
    ) -> None:
        """Implementation of a 2N point Lagrange interpolating polynomial in 1D."""
        max_idx = field.shape[0] - 2 * N  # max index for the lower index to avoid out-of-bounds
        for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # get integer and fractional parts of the index
            # N points are -(N-1), -(N-2), ..., 0, 1, ..., N relative to the lower index
            # so we subtract N from the lower index
            offset_idx = idx[i] + offset - N
            I0 = _truncate_index(offset_idx, max_idx)
            x0 = N + offset_idx - I0

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

            if not accumulate:
                output[i] = 0

            # compute interpolated value
            for j in range(2 * N):
                output[i] += l0[j] * field[I0 + j]

    return impl


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
    """Factory function creating a function implementing 2D Lagrange polynomial interpolation on a 2N point stencil."""

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
        max_idx_0 = field.shape[0] - 2 * N  # max index for the lower index to avoid out-of-bounds
        max_idx_1 = field.shape[1] - 2 * N  # max index for the lower index to avoid out-of-bounds
        for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # get integer and fractional parts of the index
            # N points are -(N-1), -(N-2), ..., 0, 1, ..., N relative to the lower index
            # so we subtract N from the lower index
            offset_idx_0 = idx0[i] + offset0 - N
            offset_idx_1 = idx1[i] + offset1 - N
            I0 = _truncate_index(offset_idx_0, max_idx_0)
            I1 = _truncate_index(offset_idx_1, max_idx_1)
            x0 = N + offset_idx_0 - I0
            x1 = N + offset_idx_1 - I1

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

            if not accumulate:
                output[i] = 0

            # compute interpolated value
            for j in range(2 * N):
                for k in range(2 * N):
                    output[i] += l0[j] * l1[k] * field[I0 + j, I1 + k]

    return impl


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
    """Factory function creating a function implementing 3D Lagrange polynomial interpolation on a 2N point stencil."""

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
        max_idx_0 = field.shape[0] - 2 * N  # max index for the lower index to avoid out-of-bounds
        max_idx_1 = field.shape[1] - 2 * N  # max index for the lower index to avoid out-of-bounds
        max_idx_2 = field.shape[2] - 2 * N  # max index for the lower index to avoid out-of-bounds
        for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # get integer and fractional parts of the index
            # N points are -(N-1), -(N-2), ..., 0, 1, ..., N relative to the lower index
            # so we subtract N from the lower index
            offset_idx_0 = idx0[i] + offset0 - N
            offset_idx_1 = idx1[i] + offset1 - N
            offset_idx_2 = idx2[i] + offset2 - N
            I0 = _truncate_index(offset_idx_0, max_idx_0)
            I1 = _truncate_index(offset_idx_1, max_idx_1)
            I2 = _truncate_index(offset_idx_2, max_idx_2)
            x0 = N + offset_idx_0 - I0
            x1 = N + offset_idx_1 - I1
            x2 = N + offset_idx_2 - I2

            # compute the Lagrange basis polynomials
            # ∏_{j=0,1,...,2N-1 j!=k} (x0 - j) / (k - j)
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

            if not accumulate:
                output[i] = 0

            # compute interpolated value
            for j in range(2 * N):
                for k in range(2 * N):
                    for l in range(2 * N):  # noqa: E741
                        output[i] += l0[j] * l1[k] * l2[l] * field[I0 + j, I1 + k, I2 + l]

    return impl


# aliases
#: linear interpolation is a special case of 1D Lagrange interpolation with N=1
linear_interpolation_factory = functools.partial(lagrange2N_1D_factory, N=1)
#: bilinear interpolation is a special case of 2D Lagrange interpolation with N=1
bilinear_interpolation_factory = functools.partial(lagrange2N_2D_factory, N=1)
#: trilinear interpolation is a special case of 3D Lagrange interpolation with N=1
trilinear_interpolation_factory = functools.partial(lagrange2N_3D_factory, N=1)
#: cubic interpolation is a special case of 1D Lagrange interpolation with N=2
cubic_interpolation_factory = functools.partial(lagrange2N_1D_factory, N=2)
#: bicubic interpolation is a special case of 2D Lagrange interpolation with N=2
bicubic_interpolation_factory = functools.partial(lagrange2N_2D_factory, N=2)
#: tricubic interpolation is a special case of 3D Lagrange interpolation with N=2
tricubic_interpolation_factory = functools.partial(lagrange2N_3D_factory, N=2)
