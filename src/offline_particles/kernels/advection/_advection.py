"""Implementation of particle advection kernels.

These kernels operate on a velocity field and a scaling field to compute the advection of particles.
The scaling field is either a metric (inverse length) or a grid spacing, and thus is
multiplied with or divided from the velocity in the advection calculation.

There are nine different advection kernel combinations, depending on the dimensionality of the velocity and scaling fields.
We explicitly construct and numba jit compile each of these combinations to ensure optimal performance.
"""

import numba
import numpy as np
import numpy.typing as npt

from ...spatial_arrays import ArrayAxis
from .._kernels import (
    BoundKernel,
    FieldDataDeclaration,
    FieldDataType,
    ParticleKernel,
    ParticlePropertiesType,
    ParticlePropertyDeclaration,
    ScalarsType,
)
from ..input_declarations import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from ..interpolation import lagrange2N_mapped_particle_factory
from ..layout_validators import ordering_validator_factory
from ..status import INACTIVE_FLAG

__all__ = ["advection_particle_kernel_factory", "construct_advection_kernel"]


def _advection_1D_1D_factory(velocity_interpolator, scaling_interpolator, N: int, metric: bool):
    r"""Create a particle advection implementation for 1D velocity and 1D scaling fields.

    Parameters
    ----------
    velocity_interpolator : Callable
        A 1D velocity field interpolator.
    scaling_interpolator : Callable
        A 1D scaling factor interpolator.
    N : int
        The half-width of the interpolation stencil.
    metric : bool
        Whether the scaling factor should be interpreted as a metric (multiplied by the velocity) or
        as a grid spacing (divided from the velocity) in the advection calculation.

    Returns
    -------
    Callable
        Aparticle advection implementation.

    Notes
    -----
    If `metric` is True, then :math:`\text{idx_tendency} += \text{velocity} \times \text{scaling}`.
    If `metric` is False, then :math:`\text{idx_tendency} += \text{velocity} / \text{scaling}`.
    """

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def advection_1D_1D(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        idx_tendency: npt.NDArray[np.float64],
        velocity_array: npt.NDArray[np.inexact],
        velocity_offset: float,
        scaling_array: npt.NDArray[np.inexact],
        scaling_offset: float,
    ) -> None:
        velocity_max_idx = velocity_array.shape[0] - 2 * N
        scaling_max_idx = scaling_array.shape[0] - 2 * N
        if velocity_max_idx < 0 or scaling_max_idx < 0:
            raise ValueError(
                "Field arrays must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            velocity = velocity_interpolator(
                zidx[i], yidx[i], xidx[i], velocity_array, velocity_offset, velocity_max_idx
            )
            scaling = scaling_interpolator(zidx[i], yidx[i], xidx[i], scaling_array, scaling_offset, scaling_max_idx)
            if metric:
                idx_tendency[i] += velocity * scaling
            else:
                idx_tendency[i] += velocity / scaling

    return advection_1D_1D


def _advection_1D_2D_factory(velocity_interpolator, scaling_interpolator, N: int, metric: bool):
    r"""Create a particle advection implementation for 1D velocity and 2D scaling fields.

    Parameters
    ----------
    velocity_interpolator : Callable
        A 1D velocity field interpolator.
    scaling_interpolator : Callable
        A 2D scaling factor interpolator.
    N : int
        The half-width of the interpolation stencil.
    metric : bool
        Whether the scaling factor should be interpreted as a metric (multiplied by the velocity) or
        as a grid spacing (divided from the velocity) in the advection calculation.

    Returns
    -------
    Callable
        A particle advection implementation.

    Notes
    -----
    If `metric` is True, then :math:`\text{idx_tendency} += \text{velocity} \times \text{scaling}`.
    If `metric` is False, then :math:`\text{idx_tendency} += \text{velocity} / \text{scaling}`.
    """

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def advection_1D_2D(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        idx_tendency: npt.NDArray[np.float64],
        velocity_array: npt.NDArray[np.inexact],
        velocity_offset: float,
        scaling_array: npt.NDArray[np.inexact],
        scaling_offset_0: float,
        scaling_offset_1: float,
    ) -> None:
        velocity_max_idx = velocity_array.shape[0] - 2 * N
        scaling_max_idx_0 = scaling_array.shape[0] - 2 * N
        scaling_max_idx_1 = scaling_array.shape[1] - 2 * N
        if velocity_max_idx < 0 or scaling_max_idx_0 < 0 or scaling_max_idx_1 < 0:
            raise ValueError(
                "Field arrays must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            velocity = velocity_interpolator(
                zidx[i], yidx[i], xidx[i], velocity_array, velocity_offset, velocity_max_idx
            )
            scaling = scaling_interpolator(
                zidx[i],
                yidx[i],
                xidx[i],
                scaling_array,
                scaling_offset_0,
                scaling_offset_1,
                scaling_max_idx_0,
                scaling_max_idx_1,
            )
            if metric:
                idx_tendency[i] += velocity * scaling
            else:
                idx_tendency[i] += velocity / scaling

    return advection_1D_2D


def _advection_1D_3D_factory(velocity_interpolator, scaling_interpolator, N: int, metric: bool):
    r"""Create a particle advection implementation for 1D velocity and 3D scaling fields.

    Parameters
    ----------
    velocity_interpolator : Callable
        A 1D velocity field interpolator.
    scaling_interpolator : Callable
        A 3D scaling factor interpolator.
    N : int
        The half-width of the interpolation stencil.
    metric : bool
        Whether the scaling factor should be interpreted as a metric (multiplied by the velocity) or
        as a grid spacing (divided from the velocity) in the advection calculation.

    Returns
    -------
    Callable
        A particle advection implementation.

    Notes
    -----
    If `metric` is True, then :math:`\text{idx_tendency} += \text{velocity} \times \text{scaling}`.
    If `metric` is False, then :math:`\text{idx_tendency} += \text{velocity} / \text{scaling}`.
    """

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def advection_1D_3D(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        idx_tendency: npt.NDArray[np.float64],
        velocity_array: npt.NDArray[np.inexact],
        velocity_offset: float,
        scaling_array: npt.NDArray[np.inexact],
        scaling_offset_0: float,
        scaling_offset_1: float,
        scaling_offset_2: float,
    ) -> None:
        velocity_max_idx = velocity_array.shape[0] - 2 * N
        scaling_max_idx_0 = scaling_array.shape[0] - 2 * N
        scaling_max_idx_1 = scaling_array.shape[1] - 2 * N
        scaling_max_idx_2 = scaling_array.shape[2] - 2 * N
        if velocity_max_idx < 0 or scaling_max_idx_0 < 0 or scaling_max_idx_1 < 0 or scaling_max_idx_2 < 0:
            raise ValueError(
                "Field arrays must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            velocity = velocity_interpolator(
                zidx[i], yidx[i], xidx[i], velocity_array, velocity_offset, velocity_max_idx
            )
            scaling = scaling_interpolator(
                zidx[i],
                yidx[i],
                xidx[i],
                scaling_array,
                scaling_offset_0,
                scaling_offset_1,
                scaling_offset_2,
                scaling_max_idx_0,
                scaling_max_idx_1,
                scaling_max_idx_2,
            )
            if metric:
                idx_tendency[i] += velocity * scaling
            else:
                idx_tendency[i] += velocity / scaling

    return advection_1D_3D


def _advection_2D_1D_factory(velocity_interpolator, scaling_interpolator, N: int, metric: bool):
    r"""Create a particle advection implementation for 2D velocity and 1D scaling fields.

    Parameters
    ----------
    velocity_interpolator : Callable
        A 2D velocity field interpolator.
    scaling_interpolator : Callable
        A 1D scaling factor interpolator.
    N : int
        The half-width of the interpolation stencil.
    metric : bool
        Whether the scaling factor should be interpreted as a metric (multiplied by the velocity) or
        as a grid spacing (divided from the velocity) in the advection calculation.

    Returns
    -------
    Callable
        A particle advection implementation.

    Notes
    -----
    If `metric` is True, then :math:`\text{idx_tendency} += \text{velocity} \times \text{scaling}`.
    If `metric` is False, then :math:`\text{idx_tendency} += \text{velocity} / \text{scaling}`.
    """

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def advection_2D_1D(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        idx_tendency: npt.NDArray[np.float64],
        velocity_array: npt.NDArray[np.inexact],
        velocity_offset_0: float,
        velocity_offset_1: float,
        scaling_array: npt.NDArray[np.inexact],
        scaling_offset: float,
    ) -> None:
        velocity_max_idx_0 = velocity_array.shape[0] - 2 * N
        velocity_max_idx_1 = velocity_array.shape[1] - 2 * N
        scaling_max_idx = scaling_array.shape[0] - 2 * N
        if velocity_max_idx_0 < 0 or velocity_max_idx_1 < 0 or scaling_max_idx < 0:
            raise ValueError(
                "Field arrays must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            velocity = velocity_interpolator(
                zidx[i],
                yidx[i],
                xidx[i],
                velocity_array,
                velocity_offset_0,
                velocity_offset_1,
                velocity_max_idx_0,
                velocity_max_idx_1,
            )
            scaling = scaling_interpolator(zidx[i], yidx[i], xidx[i], scaling_array, scaling_offset, scaling_max_idx)
            if metric:
                idx_tendency[i] += velocity * scaling
            else:
                idx_tendency[i] += velocity / scaling

    return advection_2D_1D


def _advection_2D_2D_factory(velocity_interpolator, scaling_interpolator, N: int, metric: bool):
    r"""Create a particle advection implementation for 2D velocity and 2D scaling fields.

    Parameters
    ----------
    velocity_interpolator : Callable
        A 2D velocity field interpolator.
    scaling_interpolator : Callable
        A 2D scaling factor interpolator.
    N : int
        The half-width of the interpolation stencil.
    metric : bool
        Whether the scaling factor should be interpreted as a metric (multiplied by the velocity) or
        as a grid spacing (divided from the velocity) in the advection calculation.

    Returns
    -------
    Callable
        A particle advection implementation.

    Notes
    -----
    If `metric` is True, then :math:`\text{idx_tendency} += \text{velocity} \times \text{scaling}`.
    If `metric` is False, then :math:`\text{idx_tendency} += \text{velocity} / \text{scaling}`.
    """

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def advection_2D_2D(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        idx_tendency: npt.NDArray[np.float64],
        velocity_array: npt.NDArray[np.inexact],
        velocity_offset_0: float,
        velocity_offset_1: float,
        scaling_array: npt.NDArray[np.inexact],
        scaling_offset_0: float,
        scaling_offset_1: float,
    ) -> None:
        velocity_max_idx_0 = velocity_array.shape[0] - 2 * N
        velocity_max_idx_1 = velocity_array.shape[1] - 2 * N
        scaling_max_idx_0 = scaling_array.shape[0] - 2 * N
        scaling_max_idx_1 = scaling_array.shape[1] - 2 * N
        if velocity_max_idx_0 < 0 or velocity_max_idx_1 < 0 or scaling_max_idx_0 < 0 or scaling_max_idx_1 < 0:
            raise ValueError(
                "Field arrays must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            velocity = velocity_interpolator(
                zidx[i],
                yidx[i],
                xidx[i],
                velocity_array,
                velocity_offset_0,
                velocity_offset_1,
                velocity_max_idx_0,
                velocity_max_idx_1,
            )
            scaling = scaling_interpolator(
                zidx[i],
                yidx[i],
                xidx[i],
                scaling_array,
                scaling_offset_0,
                scaling_offset_1,
                scaling_max_idx_0,
                scaling_max_idx_1,
            )
            if metric:
                idx_tendency[i] += velocity * scaling
            else:
                idx_tendency[i] += velocity / scaling

    return advection_2D_2D


def _advection_2D_3D_factory(velocity_interpolator, scaling_interpolator, N: int, metric: bool):
    r"""Create a particle advection implementation for 2D velocity and 3D scaling fields.

    Parameters
    ----------
    velocity_interpolator : Callable
        A 2D velocity field interpolator.
    scaling_interpolator : Callable
        A 3D scaling factor interpolator.
    N : int
        The half-width of the interpolation stencil.
    metric : bool
        Whether the scaling factor should be interpreted as a metric (multiplied by the velocity) or
        as a grid spacing (divided from the velocity) in the advection calculation.

    Returns
    -------
    Callable
        A particle advection implementation.

    Notes
    -----
    If `metric` is True, then :math:`\text{idx_tendency} += \text{velocity} \times \text{scaling}`.
    If `metric` is False, then :math:`\text{idx_tendency} += \text{velocity} / \text{scaling}`.
    """

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def advection_2D_3D(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        idx_tendency: npt.NDArray[np.float64],
        velocity_array: npt.NDArray[np.inexact],
        velocity_offset_0: float,
        velocity_offset_1: float,
        scaling_array: npt.NDArray[np.inexact],
        scaling_offset_0: float,
        scaling_offset_1: float,
        scaling_offset_2: float,
    ) -> None:
        velocity_max_idx_0 = velocity_array.shape[0] - 2 * N
        velocity_max_idx_1 = velocity_array.shape[1] - 2 * N
        scaling_max_idx_0 = scaling_array.shape[0] - 2 * N
        scaling_max_idx_1 = scaling_array.shape[1] - 2 * N
        scaling_max_idx_2 = scaling_array.shape[2] - 2 * N
        if (
            velocity_max_idx_0 < 0
            or velocity_max_idx_1 < 0
            or scaling_max_idx_0 < 0
            or scaling_max_idx_1 < 0
            or scaling_max_idx_2 < 0
        ):
            raise ValueError(
                "Field arrays must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            velocity = velocity_interpolator(
                zidx[i],
                yidx[i],
                xidx[i],
                velocity_array,
                velocity_offset_0,
                velocity_offset_1,
                velocity_max_idx_0,
                velocity_max_idx_1,
            )
            scaling = scaling_interpolator(
                zidx[i],
                yidx[i],
                xidx[i],
                scaling_array,
                scaling_offset_0,
                scaling_offset_1,
                scaling_offset_2,
                scaling_max_idx_0,
                scaling_max_idx_1,
                scaling_max_idx_2,
            )
            if metric:
                idx_tendency[i] += velocity * scaling
            else:
                idx_tendency[i] += velocity / scaling

    return advection_2D_3D


def _advection_3D_1D_factory(velocity_interpolator, scaling_interpolator, N: int, metric: bool):
    r"""Create a particle advection implementation for 3D velocity and 1D scaling fields.

    Parameters
    ----------
    velocity_interpolator : Callable
        A 3D velocity field interpolator.
    scaling_interpolator : Callable
        A 1D scaling factor interpolator.
    N : int
        The half-width of the interpolation stencil.
    metric : bool
        Whether the scaling factor should be interpreted as a metric (multiplied by the velocity) or
        as a grid spacing (divided from the velocity) in the advection calculation.

    Returns
    -------
    Callable
        A particle advection implementation.

    Notes
    -----
    If `metric` is True, then :math:`\text{idx_tendency} += \text{velocity} \times \text{scaling}`.
    If `metric` is False, then :math:`\text{idx_tendency} += \text{velocity} / \text{scaling}`.
    """

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def advection_3D_1D(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        idx_tendency: npt.NDArray[np.float64],
        velocity_array: npt.NDArray[np.inexact],
        velocity_offset_0: float,
        velocity_offset_1: float,
        velocity_offset_2: float,
        scaling_array: npt.NDArray[np.inexact],
        scaling_offset: float,
    ) -> None:
        velocity_max_idx_0 = velocity_array.shape[0] - 2 * N
        velocity_max_idx_1 = velocity_array.shape[1] - 2 * N
        velocity_max_idx_2 = velocity_array.shape[2] - 2 * N
        scaling_max_idx = scaling_array.shape[0] - 2 * N
        if velocity_max_idx_0 < 0 or velocity_max_idx_1 < 0 or velocity_max_idx_2 < 0 or scaling_max_idx < 0:
            raise ValueError(
                "Field arrays must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            velocity = velocity_interpolator(
                zidx[i],
                yidx[i],
                xidx[i],
                velocity_array,
                velocity_offset_0,
                velocity_offset_1,
                velocity_offset_2,
                velocity_max_idx_0,
                velocity_max_idx_1,
                velocity_max_idx_2,
            )
            scaling = scaling_interpolator(zidx[i], yidx[i], xidx[i], scaling_array, scaling_offset, scaling_max_idx)
            if metric:
                idx_tendency[i] += velocity * scaling
            else:
                idx_tendency[i] += velocity / scaling

    return advection_3D_1D


def _advection_3D_2D_factory(velocity_interpolator, scaling_interpolator, N: int, metric: bool):
    r"""Create a particle advection implementation for 3D velocity and 2D scaling fields.

    Parameters
    ----------
    velocity_interpolator : Callable
        A 3D velocity field interpolator.
    scaling_interpolator : Callable
        A 2D scaling factor interpolator.
    N : int
        The half-width of the interpolation stencil.
    metric : bool
        Whether the scaling factor should be interpreted as a metric (multiplied by the velocity) or
        as a grid spacing (divided from the velocity) in the advection calculation.

    Returns
    -------
    Callable
        A particle advection implementation.

    Notes
    -----
    If `metric` is True, then :math:`\text{idx_tendency} += \text{velocity} \times \text{scaling}`.
    If `metric` is False, then :math:`\text{idx_tendency} += \text{velocity} / \text{scaling}`.
    """

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def advection_3D_2D(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        idx_tendency: npt.NDArray[np.float64],
        velocity_array: npt.NDArray[np.inexact],
        velocity_offset_0: float,
        velocity_offset_1: float,
        velocity_offset_2: float,
        scaling_array: npt.NDArray[np.inexact],
        scaling_offset_0: float,
        scaling_offset_1: float,
    ) -> None:
        velocity_max_idx_0 = velocity_array.shape[0] - 2 * N
        velocity_max_idx_1 = velocity_array.shape[1] - 2 * N
        velocity_max_idx_2 = velocity_array.shape[2] - 2 * N
        scaling_max_idx_0 = scaling_array.shape[0] - 2 * N
        scaling_max_idx_1 = scaling_array.shape[1] - 2 * N
        if (
            velocity_max_idx_0 < 0
            or velocity_max_idx_1 < 0
            or velocity_max_idx_2 < 0
            or scaling_max_idx_0 < 0
            or scaling_max_idx_1 < 0
        ):
            raise ValueError(
                "Field arrays must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            velocity = velocity_interpolator(
                zidx[i],
                yidx[i],
                xidx[i],
                velocity_array,
                velocity_offset_0,
                velocity_offset_1,
                velocity_offset_2,
                velocity_max_idx_0,
                velocity_max_idx_1,
                velocity_max_idx_2,
            )
            scaling = scaling_interpolator(
                zidx[i],
                yidx[i],
                xidx[i],
                scaling_array,
                scaling_offset_0,
                scaling_offset_1,
                scaling_max_idx_0,
                scaling_max_idx_1,
            )
            if metric:
                idx_tendency[i] += velocity * scaling
            else:
                idx_tendency[i] += velocity / scaling

    return advection_3D_2D


def _advection_3D_3D_factory(velocity_interpolator, scaling_interpolator, N: int, metric: bool):
    r"""Create a particle advection implementation for 3D velocity and 3D scaling fields.

    Parameters
    ----------
    velocity_interpolator : Callable
        A 3D velocity field interpolator.
    scaling_interpolator : Callable
        A 3D scaling factor interpolator.
    N : int
        The half-width of the interpolation stencil.
    metric : bool
        Whether the scaling factor should be interpreted as a metric (multiplied by the velocity) or
        as a grid spacing (divided from the velocity) in the advection calculation.

    Returns
    -------
    Callable
        A particle advection implementation.

    Notes
    -----
    If `metric` is True, then :math:`\text{idx_tendency} += \text{velocity} \times \text{scaling}`.
    If `metric` is False, then :math:`\text{idx_tendency} += \text{velocity} / \text{scaling}`.
    """

    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def advection_3D_3D(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        idx_tendency: npt.NDArray[np.float64],
        velocity_array: npt.NDArray[np.inexact],
        velocity_offset_0: float,
        velocity_offset_1: float,
        velocity_offset_2: float,
        scaling_array: npt.NDArray[np.inexact],
        scaling_offset_0: float,
        scaling_offset_1: float,
        scaling_offset_2: float,
    ) -> None:
        velocity_max_idx_0 = velocity_array.shape[0] - 2 * N
        velocity_max_idx_1 = velocity_array.shape[1] - 2 * N
        velocity_max_idx_2 = velocity_array.shape[2] - 2 * N
        scaling_max_idx_0 = scaling_array.shape[0] - 2 * N
        scaling_max_idx_1 = scaling_array.shape[1] - 2 * N
        scaling_max_idx_2 = scaling_array.shape[2] - 2 * N
        if (
            velocity_max_idx_0 < 0
            or velocity_max_idx_1 < 0
            or velocity_max_idx_2 < 0
            or scaling_max_idx_0 < 0
            or scaling_max_idx_1 < 0
            or scaling_max_idx_2 < 0
        ):
            raise ValueError(
                "Field arrays must have at least 2N points in each dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.size):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            velocity = velocity_interpolator(
                zidx[i],
                yidx[i],
                xidx[i],
                velocity_array,
                velocity_offset_0,
                velocity_offset_1,
                velocity_offset_2,
                velocity_max_idx_0,
                velocity_max_idx_1,
                velocity_max_idx_2,
            )
            scaling = scaling_interpolator(
                zidx[i],
                yidx[i],
                xidx[i],
                scaling_array,
                scaling_offset_0,
                scaling_offset_1,
                scaling_offset_2,
                scaling_max_idx_0,
                scaling_max_idx_1,
                scaling_max_idx_2,
            )
            if metric:
                idx_tendency[i] += velocity * scaling
            else:
                idx_tendency[i] += velocity / scaling

    return advection_3D_3D


def advection_particle_kernel_factory(
    velocity_dim_ordering: tuple[ArrayAxis, ...],
    scaling_dim_ordering: tuple[ArrayAxis, ...],
    N: int = 1,
    metric: bool = True,
) -> ParticleKernel:
    """Create a particle advection kernel for the given velocity and spatial dimension orderings.

    Parameters
    ----------
    velocity_dim_ordering : tuple[ArrayAxis, ...]
        The ordering of dimensions in the velocity field.
    scaling_dim_ordering : tuple[ArrayAxis, ...]
        The ordering of dimensions in the scaling field.
    N : int, optional
        The half-width of the interpolation stencil. Defaults to 1 (linear interpolation).
    metric : bool, optional
        Whether to interpret the scaling as a metric (multiply velocity) or grid spacing (divided from the velocity) in the advection calculation.
        Default is True.

    Returns
    -------
    ParticleKernel
        A particle kernel implementing advection.

    Raises
    ------
    ValueError
        If the dimensionality of the velocity or spatial fields is not supported.
        If either dim_ordering is an invalid argument to :function:`lagrange2N_mapped_particle_factory`.
    """
    velocity_interpolator = lagrange2N_mapped_particle_factory(velocity_dim_ordering, N)
    scaling_interpolator = lagrange2N_mapped_particle_factory(scaling_dim_ordering, N)

    match len(velocity_dim_ordering), len(scaling_dim_ordering):
        case 1, 1:
            impl = _advection_1D_1D_factory(velocity_interpolator, scaling_interpolator, N, metric)
        case 1, 2:
            impl = _advection_1D_2D_factory(velocity_interpolator, scaling_interpolator, N, metric)
        case 1, 3:
            impl = _advection_1D_3D_factory(velocity_interpolator, scaling_interpolator, N, metric)
        case 2, 1:
            impl = _advection_2D_1D_factory(velocity_interpolator, scaling_interpolator, N, metric)
        case 2, 2:
            impl = _advection_2D_2D_factory(velocity_interpolator, scaling_interpolator, N, metric)
        case 2, 3:
            impl = _advection_2D_3D_factory(velocity_interpolator, scaling_interpolator, N, metric)
        case 3, 1:
            impl = _advection_3D_1D_factory(velocity_interpolator, scaling_interpolator, N, metric)
        case 3, 2:
            impl = _advection_3D_2D_factory(velocity_interpolator, scaling_interpolator, N, metric)
        case 3, 3:
            impl = _advection_3D_3D_factory(velocity_interpolator, scaling_interpolator, N, metric)
        case _:
            raise ValueError(
                "Unsupported dimensionality of velocity or scaling fields for particle advection. 1,2, and 3 dimensions are supported for both velocity and scaling fields."
            )

    def advection_kernel_function(
        particle_properties: ParticlePropertiesType, scalars: ScalarsType, field_data: FieldDataType
    ) -> None:
        velocity_array = field_data["velocity"].array
        velocity_offsets = field_data["velocity"].offsets
        scaling_array = field_data["scaling"].array
        scaling_offsets = field_data["scaling"].offsets
        return impl(
            particle_properties["status"],
            particle_properties["zidx"],
            particle_properties["yidx"],
            particle_properties["xidx"],
            particle_properties["idx_tendency"],
            velocity_array,
            *velocity_offsets,
            scaling_array,
            *scaling_offsets,
        )

    velocity_validator = ordering_validator_factory(velocity_dim_ordering)
    scaling_validator = ordering_validator_factory(scaling_dim_ordering)

    kernel = ParticleKernel(
        advection_kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ZIDX_DECLARATION,
            YIDX_DECLARATION,
            XIDX_DECLARATION,
            ParticlePropertyDeclaration("idx_tendency", np.float64),
        ],
        field_data=[
            FieldDataDeclaration("velocity", np.inexact, [velocity_validator]),
            FieldDataDeclaration("scaling", np.inexact, [scaling_validator]),
        ],
    )

    return kernel


def construct_advection_kernel(
    idx_tendency_binding: str,
    velocity_binding: str,
    scaling_binding: str,
    velocity_dim_ordering: tuple[ArrayAxis, ...],
    scaling_dim_ordering: tuple[ArrayAxis, ...],
    N: int = 1,
    metric: bool = True,
) -> BoundKernel:
    """Create an advection kernel for the given velocity and spatial dimension orderings.

    Parameters
    ----------
    idx_tendency_binding : str
        The binding for the index tendency particle property.
    velocity_binding : str
        The binding for the velocity field.
    scaling_binding : str
        The binding for the scaling field.
    velocity_dim_ordering : tuple[ArrayAxis, ...]
        The ordering of dimensions in the velocity field.
    scaling_dim_ordering : tuple[ArrayAxis, ...]
        The ordering of dimensions in the scaling field.
    N : int, optional
        The half-width of the interpolation stencil. Defaults to 1 (linear interpolation).
    metric : bool, optional
        Whether to interpret the scaling as a metric (multiply velocity) or grid spacing (divided from the velocity) in the advection calculation.
        Default is True.

    Returns
    -------
    BoundKernel
        A bound kernel implementing advection.

    Raises
    ------
    ValueError
        If the dimensionality of the velocity or spatial fields is not supported or the dim_ordering arguments are invalid.
        From :function:`advection_particle_kernel_factory`.
    """
    kernel = advection_particle_kernel_factory(velocity_dim_ordering, scaling_dim_ordering, N, metric)
    return BoundKernel(
        kernel,
        particle_property_bindings={"idx_tendency": idx_tendency_binding},
        field_data_bindings={"velocity": velocity_binding, "scaling": scaling_binding},
    )
