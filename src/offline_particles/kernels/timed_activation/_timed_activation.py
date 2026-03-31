import numba
import numpy as np
import numpy.typing as npt

from .._kernels import FieldDataType, ParticlePropertiesType, ScalarsType
from ..status import INACTIVE_FLAG, Status

type T = np.float64 | np.datetime64


@numba.njit(parallel=True, nogil=True, fastmath=True)
def _activate_released_particles(
    status: npt.NDArray[np.uint8],
    release_time: npt.NDArray[T],
    time: T,
    dt: np.float64,
) -> None:
    for i in numba.prange(status.shape[0]):  # ty: ignore[not-iterable]
        if status[i] != Status.PRE_RELEASE:
            continue
        if dt > 0 and release_time[i] <= time or dt < 0 and release_time[i] > time:
            status[i] = Status.NORMAL


@numba.njit(parallel=True, nogil=True, fastmath=True)
def _deactivate_retired_particles(
    status: npt.NDArray[np.uint8],
    retirement_time: npt.NDArray[T],
    time: T,
    dt: np.float64,
) -> None:
    for i in numba.prange(status.shape[0]):  # ty: ignore[not-iterable]
        if status[i] & INACTIVE_FLAG:
            continue
        if dt > 0 and retirement_time[i] <= time or dt < 0 and retirement_time[i] > time:
            status[i] = Status.POST_RETIREMENT


# kernel functions
def activate_released_particles(
    particle_properties: ParticlePropertiesType, scalars: ScalarsType, fields: FieldDataType
) -> None:
    _activate_released_particles(
        particle_properties["status"],
        particle_properties["release_time"],
        scalars["_time"],
        scalars["_dt"],
    )


def deactivate_retired_particles(
    particle_properties: ParticlePropertiesType, scalars: ScalarsType, fields: FieldDataType
) -> None:
    _deactivate_retired_particles(
        particle_properties["status"],
        particle_properties["retirement_time"],
        scalars["_time"],
        scalars["_dt"],
    )
