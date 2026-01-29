"""Kernel functions implementing basic particle operations."""

import numba
import numpy as np
import numpy.typing as npt

from .._kernels import FieldDataType, ParticlePropertiesType, ScalarsType
from ..status import INACTIVE_FLAG


@numba.njit(parallel=True, nogil=True, fastmath=True)
def _copy_property(
    status: npt.NDArray[np.uint8],
    source: npt.NDArray[np.generic],
    destination: npt.NDArray[np.generic],
) -> None:
    for i in numba.prange(status.shape[0]):  # ty: ignore[not-iterable]
        if status[i] & INACTIVE_FLAG:
            continue
        destination[i] = source[i]


@numba.njit(parallel=True, nogil=True, fastmath=True)
def _add_property(
    status: npt.NDArray[np.uint8],
    source: npt.NDArray[np.generic],
    destination: npt.NDArray[np.generic],
) -> None:
    for i in numba.prange(status.shape[0]):  # ty: ignore[not-iterable]
        if status[i] & INACTIVE_FLAG:
            continue
        destination[i] += source[i]


@numba.njit(parallel=True, nogil=True, fastmath=True)
def _subtract_property(
    status: npt.NDArray[np.uint8],
    source: npt.NDArray[np.generic],
    destination: npt.NDArray[np.generic],
) -> None:
    for i in numba.prange(status.shape[0]):  # ty: ignore[not-iterable]
        if status[i] & INACTIVE_FLAG:
            continue
        destination[i] -= source[i]


@numba.njit(parallel=True, nogil=True, fastmath=True)
def _multiply_property(
    status: npt.NDArray[np.uint8],
    source: npt.NDArray[np.generic],
    destination: npt.NDArray[np.generic],
) -> None:
    for i in numba.prange(status.shape[0]):  # ty: ignore[not-iterable]
        if status[i] & INACTIVE_FLAG:
            continue
        destination[i] *= source[i]


@numba.njit(parallel=True, nogil=True)
def _divide_property(
    status: npt.NDArray[np.uint8],
    source: npt.NDArray[np.generic],
    destination: npt.NDArray[np.generic],
) -> None:
    for i in numba.prange(status.shape[0]):  # ty: ignore[not-iterable]
        if status[i] & INACTIVE_FLAG:
            continue
        destination[i] /= source[i]


def copy_property(particle_properties: ParticlePropertiesType, fields: FieldDataType, scalars: ScalarsType) -> None:
    _copy_property(
        particle_properties["status"],
        particle_properties["source"],
        particle_properties["destination"],
    )


def add_property(particle_properties: ParticlePropertiesType, fields: FieldDataType, scalars: ScalarsType) -> None:
    _add_property(
        particle_properties["status"],
        particle_properties["source"],
        particle_properties["destination"],
    )


def subtract_property(particle_properties: ParticlePropertiesType, fields: FieldDataType, scalars: ScalarsType) -> None:
    _subtract_property(
        particle_properties["status"],
        particle_properties["source"],
        particle_properties["destination"],
    )


def multiply_property(particle_properties: ParticlePropertiesType, fields: FieldDataType, scalars: ScalarsType) -> None:
    _multiply_property(
        particle_properties["status"],
        particle_properties["source"],
        particle_properties["destination"],
    )


def divide_property(particle_properties: ParticlePropertiesType, fields: FieldDataType, scalars: ScalarsType) -> None:
    _divide_property(
        particle_properties["status"],
        particle_properties["source"],
        particle_properties["destination"],
    )
