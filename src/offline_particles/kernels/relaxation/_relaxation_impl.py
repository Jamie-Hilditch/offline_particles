"""Implementations of linear and quadratic relaxation tendencies for particle properties."""

from functools import lru_cache
from typing import Callable

import numba
import numpy as np
import numpy.typing as npt

from ..interpolation import (
    lagrange2N_1D_particle_factory,
    lagrange2N_2D_particle_factory,
    lagrange2N_3D_particle_factory,
)
from ..status import INACTIVE_FLAG

__all__ = [
    "_linear_relaxation_constant_coefficient_constant_target",
    "_linear_relaxation_constant_coefficient_property_target",
    "_linear_relaxation_constant_coefficient_scalar_target",
    "_linear_relaxation_constant_coefficient_1D_field_target",
    "_linear_relaxation_constant_coefficient_2D_field_target",
    "_linear_relaxation_constant_coefficient_3D_field_target",
    "_linear_relaxation_property_coefficient_constant_target",
    "_linear_relaxation_property_coefficient_property_target",
    "_linear_relaxation_property_coefficient_scalar_target",
    "_linear_relaxation_property_coefficient_1D_field_target",
    "_linear_relaxation_property_coefficient_2D_field_target",
    "_linear_relaxation_property_coefficient_3D_field_target",
    "_linear_relaxation_scalar_coefficient_constant_target",
    "_linear_relaxation_scalar_coefficient_property_target",
    "_linear_relaxation_scalar_coefficient_scalar_target",
    "_linear_relaxation_scalar_coefficient_1D_field_target",
    "_linear_relaxation_scalar_coefficient_2D_field_target",
    "_linear_relaxation_scalar_coefficient_3D_field_target",
    "_quadratic_relaxation_constant_coefficient_constant_target",
    "_quadratic_relaxation_constant_coefficient_property_target",
    "_quadratic_relaxation_constant_coefficient_scalar_target",
    "_quadratic_relaxation_constant_coefficient_1D_field_target",
    "_quadratic_relaxation_constant_coefficient_2D_field_target",
    "_quadratic_relaxation_constant_coefficient_3D_field_target",
    "_quadratic_relaxation_property_coefficient_constant_target",
    "_quadratic_relaxation_property_coefficient_property_target",
    "_quadratic_relaxation_property_coefficient_scalar_target",
    "_quadratic_relaxation_property_coefficient_1D_field_target",
    "_quadratic_relaxation_property_coefficient_2D_field_target",
    "_quadratic_relaxation_property_coefficient_3D_field_target",
    "_quadratic_relaxation_scalar_coefficient_constant_target",
    "_quadratic_relaxation_scalar_coefficient_property_target",
    "_quadratic_relaxation_scalar_coefficient_scalar_target",
    "_quadratic_relaxation_scalar_coefficient_1D_field_target",
    "_quadratic_relaxation_scalar_coefficient_2D_field_target",
    "_quadratic_relaxation_scalar_coefficient_3D_field_target",
]


@lru_cache(maxsize=None)
def _linear_relaxation_constant_coefficient_constant_target(
    relaxation_coefficient: np.inexact,
    target: np.inexact,
) -> Callable[[npt.NDArray[np.uint8], npt.NDArray[np.inexact], npt.NDArray[np.inexact]], None]:
    """Construct a function to apply linear relaxation with constant coefficient and target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            dprop[i] -= relaxation_coefficient * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_constant_coefficient_property_target(
    relaxation_coefficient: np.inexact,
) -> Callable[[npt.NDArray[np.uint8], npt.NDArray[np.inexact], npt.NDArray[np.inexact], npt.NDArray[np.inexact]], None]:
    """Construct a function to apply linear relaxation with constant coefficient and property target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        target: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            dprop[i] -= relaxation_coefficient * (prop[i] - target[i])

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_constant_coefficient_scalar_target(
    relaxation_coefficient: np.inexact,
) -> Callable[[npt.NDArray[np.uint8], npt.NDArray[np.inexact], npt.NDArray[np.inexact], np.generic], None]:
    """Construct a function to apply linear relaxation with constant coefficient and scalar target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target: np.generic,
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            dprop[i] -= relaxation_coefficient * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_constant_coefficient_1D_field_target(
    relaxation_coefficient: np.inexact,
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        float,
    ],
    None,
]:
    """Construct a function to apply linear relaxation with constant coefficient and 1D field target.

    Parameters
    ----------
    relaxation_coefficient: The constant relaxation coefficient to apply.
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to linear interpolation.
    """
    interpolator = lagrange2N_1D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target_array: npt.NDArray[np.inexact],
        offset: float,
    ) -> None:
        # max index for the lower index to avoid out-of-bounds
        max_idx = target_array.shape[0] - 2 * interpolation_half_width
        if max_idx < 0:
            raise ValueError(
                "Target array must have at least 2N points in the relevant dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx = idx[i] + offset
            target = interpolator(target_array, offset_idx, max_idx)

            dprop[i] -= relaxation_coefficient * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_constant_coefficient_2D_field_target(
    relaxation_coefficient: np.inexact,
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        float,
        float,
    ],
    None,
]:
    """Construct a function to apply linear relaxation with constant coefficient and 2D field target.

    Parameters
    ----------
    relaxation_coefficient: The constant relaxation coefficient to apply.
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to bilinear interpolation.
    """
    interpolator = lagrange2N_2D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
    ) -> None:
        # set the max indices to avoid out-of-bounds memory access
        max_idx_0 = target_array.shape[0] - 2 * interpolation_half_width
        max_idx_1 = target_array.shape[1] - 2 * interpolation_half_width
        if max_idx_0 < 0 or max_idx_1 < 0:
            raise ValueError(
                "Target array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            target = interpolator(target_array, offset_idx_0, offset_idx_1, max_idx_0, max_idx_1)

            dprop[i] -= relaxation_coefficient * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_constant_coefficient_3D_field_target(
    relaxation_coefficient: np.inexact,
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        float,
        float,
        float,
    ],
    None,
]:
    """Construct a function to apply linear relaxation with constant coefficient and 3D field target.

    Parameters
    ----------
    relaxation_coefficient: The constant relaxation coefficient to apply.
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to trilinear interpolation.
    """
    interpolator = lagrange2N_3D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        idx2: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
        offset2: float,
    ) -> None:
        # set the max indices to avoid out-of-bounds memory access
        max_idx_0 = target_array.shape[0] - 2 * interpolation_half_width
        max_idx_1 = target_array.shape[1] - 2 * interpolation_half_width
        max_idx_2 = target_array.shape[2] - 2 * interpolation_half_width
        if max_idx_0 < 0 or max_idx_1 < 0 or max_idx_2 < 0:
            raise ValueError(
                "Target array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            offset_idx_2 = idx2[i] + offset2
            target = interpolator(
                target_array, offset_idx_0, offset_idx_1, offset_idx_2, max_idx_0, max_idx_1, max_idx_2
            )

            dprop[i] -= relaxation_coefficient * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_property_coefficient_constant_target(
    target: np.inexact,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
    ],
    None,
]:
    """Construct a function to apply linear relaxation with property coefficient and constant target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        relaxation_coefficient: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            dprop[i] -= relaxation_coefficient[i] * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_property_coefficient_property_target() -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
    ],
    None,
]:
    """Construct a function to apply linear relaxation with property coefficient and property target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        relaxation_coefficient: npt.NDArray[np.inexact],
        target: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            dprop[i] -= relaxation_coefficient[i] * (prop[i] - target[i])

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_property_coefficient_scalar_target() -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        np.generic,
    ],
    None,
]:
    """Construct a function to apply linear relaxation with property coefficient and scalar target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        relaxation_coefficient: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target: np.generic,
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            dprop[i] -= relaxation_coefficient[i] * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_property_coefficient_1D_field_target(
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        float,
    ],
    None,
]:
    """Construct a function to apply linear relaxation with property coefficient and 1D field target.

    Parameters
    ----------
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to linear interpolation.
    """
    interpolator = lagrange2N_1D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        relaxation_coefficient: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target_array: npt.NDArray[np.inexact],
        offset: float,
    ) -> None:
        # max index for the lower index to avoid out-of-bounds
        max_idx = target_array.shape[0] - 2 * interpolation_half_width
        if max_idx < 0:
            raise ValueError(
                "Target array must have at least 2N points in the relevant dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx = idx[i] + offset
            target = interpolator(target_array, offset_idx, max_idx)

            dprop[i] -= relaxation_coefficient[i] * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_property_coefficient_2D_field_target(
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        float,
        float,
    ],
    None,
]:
    """Construct a function to apply linear relaxation with property coefficient and 2D field target.

    Parameters
    ----------
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to bilinear interpolation.
    """
    interpolator = lagrange2N_2D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        relaxation_coefficient: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
    ) -> None:
        # set the max indices to avoid out-of-bounds memory access
        max_idx_0 = target_array.shape[0] - 2 * interpolation_half_width
        max_idx_1 = target_array.shape[1] - 2 * interpolation_half_width
        if max_idx_0 < 0 or max_idx_1 < 0:
            raise ValueError(
                "Target array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            target = interpolator(target_array, offset_idx_0, offset_idx_1, max_idx_0, max_idx_1)

            dprop[i] -= relaxation_coefficient[i] * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_property_coefficient_3D_field_target(
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        float,
        float,
        float,
    ],
    None,
]:
    """Construct a function to apply linear relaxation with property coefficient and 3D field target.

    Parameters
    ----------
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to trilinear interpolation.
    """
    interpolator = lagrange2N_3D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        idx2: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        relaxation_coefficient: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
        offset2: float,
    ) -> None:
        # set the max indices to avoid out-of-bounds memory access
        max_idx_0 = target_array.shape[0] - 2 * interpolation_half_width
        max_idx_1 = target_array.shape[1] - 2 * interpolation_half_width
        max_idx_2 = target_array.shape[2] - 2 * interpolation_half_width
        if max_idx_0 < 0 or max_idx_1 < 0 or max_idx_2 < 0:
            raise ValueError(
                "Target array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            offset_idx_2 = idx2[i] + offset2
            target = interpolator(
                target_array, offset_idx_0, offset_idx_1, offset_idx_2, max_idx_0, max_idx_1, max_idx_2
            )

            dprop[i] -= relaxation_coefficient[i] * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_scalar_coefficient_constant_target(
    target: np.inexact,
) -> Callable[[npt.NDArray[np.uint8], npt.NDArray[np.inexact], npt.NDArray[np.inexact], np.generic], None]:
    """Construct a function to apply linear relaxation with scalar coefficient and constant target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        relaxation_coefficient: np.generic,
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            dprop[i] -= relaxation_coefficient * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_scalar_coefficient_property_target() -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        np.generic,
    ],
    None,
]:
    """Construct a function to apply linear relaxation with scalar coefficient and property target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        target: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        relaxation_coefficient: np.generic,
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            dprop[i] -= relaxation_coefficient * (prop[i] - target[i])

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_scalar_coefficient_scalar_target() -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        np.generic,
        np.generic,
    ],
    None,
]:
    """Construct a function to apply linear relaxation with scalar coefficient and scalar target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        relaxation_coefficient: np.generic,
        target: np.generic,
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            dprop[i] -= relaxation_coefficient * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_scalar_coefficient_1D_field_target(
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        np.generic,
        npt.NDArray[np.inexact],
        float,
    ],
    None,
]:
    """Construct a function to apply linear relaxation with scalar coefficient and 1D field target.

    Parameters
    ----------
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to linear interpolation.
    """
    interpolator = lagrange2N_1D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        relaxation_coefficient: np.generic,
        target_array: npt.NDArray[np.inexact],
        offset: float,
    ) -> None:
        # max index for the lower index to avoid out-of-bounds
        max_idx = target_array.shape[0] - 2 * interpolation_half_width
        if max_idx < 0:
            raise ValueError(
                "Target array must have at least 2N points in the relevant dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx = idx[i] + offset
            target = interpolator(target_array, offset_idx, max_idx)

            dprop[i] -= relaxation_coefficient * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_scalar_coefficient_2D_field_target(
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        np.generic,
        npt.NDArray[np.inexact],
        float,
        float,
    ],
    None,
]:
    """Construct a function to apply linear relaxation with scalar coefficient and 2D field target.

    Parameters
    ----------
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to bilinear interpolation.
    """
    interpolator = lagrange2N_2D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        relaxation_coefficient: np.generic,
        target_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
    ) -> None:
        # set the max indices to avoid out-of-bounds memory access
        max_idx_0 = target_array.shape[0] - 2 * interpolation_half_width
        max_idx_1 = target_array.shape[1] - 2 * interpolation_half_width
        if max_idx_0 < 0 or max_idx_1 < 0:
            raise ValueError(
                "Target array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            target = interpolator(target_array, offset_idx_0, offset_idx_1, max_idx_0, max_idx_1)

            dprop[i] -= relaxation_coefficient * (prop[i] - target)

    return impl


@lru_cache(maxsize=None)
def _linear_relaxation_scalar_coefficient_3D_field_target(
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        np.generic,
        npt.NDArray[np.inexact],
        float,
        float,
        float,
    ],
    None,
]:
    """Construct a function to apply linear relaxation with scalar coefficient and 3D field target.

    Parameters
    ----------
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to trilinear interpolation.
    """
    interpolator = lagrange2N_3D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        idx2: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        relaxation_coefficient: np.generic,
        target_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
        offset2: float,
    ) -> None:
        # set the max indices to avoid out-of-bounds memory access
        max_idx_0 = target_array.shape[0] - 2 * interpolation_half_width
        max_idx_1 = target_array.shape[1] - 2 * interpolation_half_width
        max_idx_2 = target_array.shape[2] - 2 * interpolation_half_width
        if max_idx_0 < 0 or max_idx_1 < 0 or max_idx_2 < 0:
            raise ValueError(
                "Target array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            offset_idx_2 = idx2[i] + offset2
            target = interpolator(
                target_array, offset_idx_0, offset_idx_1, offset_idx_2, max_idx_0, max_idx_1, max_idx_2
            )

            dprop[i] -= relaxation_coefficient * (prop[i] - target)

    return impl


########################
# Quadratic relaxation #
########################


@lru_cache(maxsize=None)
def _quadratic_relaxation_constant_coefficient_constant_target(
    relaxation_coefficient: np.inexact,
    target: np.inexact,
) -> Callable[[npt.NDArray[np.uint8], npt.NDArray[np.inexact], npt.NDArray[np.inexact]], None]:
    """Construct a function to apply quadratic relaxation with constant coefficient and target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_constant_coefficient_property_target(
    relaxation_coefficient: np.inexact,
) -> Callable[[npt.NDArray[np.uint8], npt.NDArray[np.inexact], npt.NDArray[np.inexact], npt.NDArray[np.inexact]], None]:
    """Construct a function to apply quadratic relaxation with constant coefficient and property target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        target: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            diff = prop[i] - target[i]
            dprop[i] -= relaxation_coefficient * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_constant_coefficient_scalar_target(
    relaxation_coefficient: np.inexact,
) -> Callable[[npt.NDArray[np.uint8], npt.NDArray[np.inexact], npt.NDArray[np.inexact], np.generic], None]:
    """Construct a function to apply quadratic relaxation with constant coefficient and scalar target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target: np.generic,
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_constant_coefficient_1D_field_target(
    relaxation_coefficient: np.inexact,
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        float,
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with constant coefficient and 1D field target.

    Parameters
    ----------
    relaxation_coefficient: The constant relaxation coefficient to apply.
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to linear interpolation.
    """
    interpolator = lagrange2N_1D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target_array: npt.NDArray[np.inexact],
        offset: float,
    ) -> None:
        # max index for the lower index to avoid out-of-bounds
        max_idx = target_array.shape[0] - 2 * interpolation_half_width
        if max_idx < 0:
            raise ValueError(
                "Target array must have at least 2N points in the relevant dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx = idx[i] + offset
            target = interpolator(target_array, offset_idx, max_idx)

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_constant_coefficient_2D_field_target(
    relaxation_coefficient: np.inexact,
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        float,
        float,
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with constant coefficient and 2D field target.

    Parameters
    ----------
    relaxation_coefficient: The constant relaxation coefficient to apply.
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to bilinear interpolation.
    """
    interpolator = lagrange2N_2D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
    ) -> None:
        # set the max indices to avoid out-of-bounds memory access
        max_idx_0 = target_array.shape[0] - 2 * interpolation_half_width
        max_idx_1 = target_array.shape[1] - 2 * interpolation_half_width
        if max_idx_0 < 0 or max_idx_1 < 0:
            raise ValueError(
                "Target array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            target = interpolator(target_array, offset_idx_0, offset_idx_1, max_idx_0, max_idx_1)

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_constant_coefficient_3D_field_target(
    relaxation_coefficient: np.inexact,
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        float,
        float,
        float,
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with constant coefficient and 3D field target.

    Parameters
    ----------
    relaxation_coefficient: The constant relaxation coefficient to apply.
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to trilinear interpolation.
    """
    interpolator = lagrange2N_3D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        idx2: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
        offset2: float,
    ) -> None:
        # set the max indices to avoid out-of-bounds memory access
        max_idx_0 = target_array.shape[0] - 2 * interpolation_half_width
        max_idx_1 = target_array.shape[1] - 2 * interpolation_half_width
        max_idx_2 = target_array.shape[2] - 2 * interpolation_half_width
        if max_idx_0 < 0 or max_idx_1 < 0 or max_idx_2 < 0:
            raise ValueError(
                "Target array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            offset_idx_2 = idx2[i] + offset2
            target = interpolator(
                target_array, offset_idx_0, offset_idx_1, offset_idx_2, max_idx_0, max_idx_1, max_idx_2
            )

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_property_coefficient_constant_target(
    target: np.inexact,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with property coefficient and constant target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        relaxation_coefficient: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient[i] * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_property_coefficient_property_target() -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with property coefficient and property target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        relaxation_coefficient: npt.NDArray[np.inexact],
        target: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            diff = prop[i] - target[i]
            dprop[i] -= relaxation_coefficient[i] * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_property_coefficient_scalar_target() -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        np.generic,
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with property coefficient and scalar target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        relaxation_coefficient: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target: np.generic,
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient[i] * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_property_coefficient_1D_field_target(
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        float,
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with property coefficient and 1D field target.

    Parameters
    ----------
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to linear interpolation.
    """
    interpolator = lagrange2N_1D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        relaxation_coefficient: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target_array: npt.NDArray[np.inexact],
        offset: float,
    ) -> None:
        # max index for the lower index to avoid out-of-bounds
        max_idx = target_array.shape[0] - 2 * interpolation_half_width
        if max_idx < 0:
            raise ValueError(
                "Target array must have at least 2N points in the relevant dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx = idx[i] + offset
            target = interpolator(target_array, offset_idx, max_idx)

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient[i] * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_property_coefficient_2D_field_target(
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        float,
        float,
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with property coefficient and 2D field target.

    Parameters
    ----------
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to bilinear interpolation.
    """
    interpolator = lagrange2N_2D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        relaxation_coefficient: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
    ) -> None:
        # set the max indices to avoid out-of-bounds memory access
        max_idx_0 = target_array.shape[0] - 2 * interpolation_half_width
        max_idx_1 = target_array.shape[1] - 2 * interpolation_half_width
        if max_idx_0 < 0 or max_idx_1 < 0:
            raise ValueError(
                "Target array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            target = interpolator(target_array, offset_idx_0, offset_idx_1, max_idx_0, max_idx_1)

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient[i] * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_property_coefficient_3D_field_target(
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        float,
        float,
        float,
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with property coefficient and 3D field target.

    Parameters
    ----------
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to trilinear interpolation.
    """
    interpolator = lagrange2N_3D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        idx2: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        relaxation_coefficient: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        target_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
        offset2: float,
    ) -> None:
        # set the max indices to avoid out-of-bounds memory access
        max_idx_0 = target_array.shape[0] - 2 * interpolation_half_width
        max_idx_1 = target_array.shape[1] - 2 * interpolation_half_width
        max_idx_2 = target_array.shape[2] - 2 * interpolation_half_width
        if max_idx_0 < 0 or max_idx_1 < 0 or max_idx_2 < 0:
            raise ValueError(
                "Target array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            offset_idx_2 = idx2[i] + offset2
            target = interpolator(
                target_array, offset_idx_0, offset_idx_1, offset_idx_2, max_idx_0, max_idx_1, max_idx_2
            )

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient[i] * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_scalar_coefficient_constant_target(
    target: np.inexact,
) -> Callable[[npt.NDArray[np.uint8], npt.NDArray[np.inexact], npt.NDArray[np.inexact], np.generic], None]:
    """Construct a function to apply quadratic relaxation with scalar coefficient and constant target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        relaxation_coefficient: np.generic,
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_scalar_coefficient_property_target() -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        np.generic,
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with scalar coefficient and property target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        target: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        relaxation_coefficient: np.generic,
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            diff = prop[i] - target[i]
            dprop[i] -= relaxation_coefficient * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_scalar_coefficient_scalar_target() -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        np.generic,
        np.generic,
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with scalar coefficient and scalar target."""

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        relaxation_coefficient: np.generic,
        target: np.generic,
    ) -> None:
        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_scalar_coefficient_1D_field_target(
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        np.generic,
        npt.NDArray[np.inexact],
        float,
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with scalar coefficient and 1D field target.

    Parameters
    ----------
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to linear interpolation.
    """
    interpolator = lagrange2N_1D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        relaxation_coefficient: np.generic,
        target_array: npt.NDArray[np.inexact],
        offset: float,
    ) -> None:
        # max index for the lower index to avoid out-of-bounds
        max_idx = target_array.shape[0] - 2 * interpolation_half_width
        if max_idx < 0:
            raise ValueError(
                "Target array must have at least 2N points in the relevant dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx = idx[i] + offset
            target = interpolator(target_array, offset_idx, max_idx)

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_scalar_coefficient_2D_field_target(
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        np.generic,
        npt.NDArray[np.inexact],
        float,
        float,
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with scalar coefficient and 2D field target.

    Parameters
    ----------
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to bilinear interpolation.
    """
    interpolator = lagrange2N_2D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        relaxation_coefficient: np.generic,
        target_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
    ) -> None:
        # set the max indices to avoid out-of-bounds memory access
        max_idx_0 = target_array.shape[0] - 2 * interpolation_half_width
        max_idx_1 = target_array.shape[1] - 2 * interpolation_half_width
        if max_idx_0 < 0 or max_idx_1 < 0:
            raise ValueError(
                "Target array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            target = interpolator(target_array, offset_idx_0, offset_idx_1, max_idx_0, max_idx_1)

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient * diff * np.abs(diff)

    return impl


@lru_cache(maxsize=None)
def _quadratic_relaxation_scalar_coefficient_3D_field_target(
    interpolation_half_width: int = 1,
) -> Callable[
    [
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.inexact],
        npt.NDArray[np.inexact],
        np.generic,
        npt.NDArray[np.inexact],
        float,
        float,
        float,
    ],
    None,
]:
    """Construct a function to apply quadratic relaxation with scalar coefficient and 3D field target.

    Parameters
    ----------
    interpolation_half_width: The half-width of the interpolation stencil to use for sampling the field target.
        Default is 1, which corresponds to trilinear interpolation.
    """
    interpolator = lagrange2N_3D_particle_factory(interpolation_half_width)

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def impl(
        status: npt.NDArray[np.uint8],
        idx0: npt.NDArray[np.float64],
        idx1: npt.NDArray[np.float64],
        idx2: npt.NDArray[np.float64],
        prop: npt.NDArray[np.inexact],
        dprop: npt.NDArray[np.inexact],
        relaxation_coefficient: np.generic,
        target_array: npt.NDArray[np.inexact],
        offset0: float,
        offset1: float,
        offset2: float,
    ) -> None:
        # set the max indices to avoid out-of-bounds memory access
        max_idx_0 = target_array.shape[0] - 2 * interpolation_half_width
        max_idx_1 = target_array.shape[1] - 2 * interpolation_half_width
        max_idx_2 = target_array.shape[2] - 2 * interpolation_half_width
        if max_idx_0 < 0 or max_idx_1 < 0 or max_idx_2 < 0:
            raise ValueError(
                "Target array must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # ty: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            # interpolate target value
            offset_idx_0 = idx0[i] + offset0
            offset_idx_1 = idx1[i] + offset1
            offset_idx_2 = idx2[i] + offset2
            target = interpolator(
                target_array, offset_idx_0, offset_idx_1, offset_idx_2, max_idx_0, max_idx_1, max_idx_2
            )

            diff = prop[i] - target
            dprop[i] -= relaxation_coefficient * diff * np.abs(diff)

    return impl
