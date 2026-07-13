import enum
from typing import overload

import numba
import numpy as np
import numpy.typing as npt

from .._kernels import BoundKernel, FieldDataType, ParticleKernel, ParticlePropertiesType, ScalarsType, kernel_function
from ..input_declarations import STATUS_DECLARATION

__all__ = [
    "INACTIVE_FLAG",
    "FieldDataType",
    "ParticlePropertiesType",
    "ScalarsType",
    "Status",
    "construct_initialise_status_kernel",
    "is_active",
    "is_inactive",
]


class Status(enum.IntEnum):
    """Enumeration of particle status codes."""

    # bit flag for active/inactive particles; reserve the final bit for the inactive flag
    INACTIVE = 1 << 7

    # normal state
    NORMAL = 0

    # error states
    NONFINITE = 1 | INACTIVE
    OUT_OF_DOMAIN = 2 | INACTIVE
    BELOW_BOTTOM = 3 | INACTIVE
    ABOVE_SURFACE = 4 | INACTIVE

    # reserved for multistep initialization
    MULTISTEP_1 = 10
    MULTISTEP_2 = 11

    # recurring initialisation phase (both sim-start and mid-simulation, e.g. timed release)
    INITIALISING = 19 | INACTIVE

    # timed releases and retirements
    PRE_RELEASE = 20 | INACTIVE
    POST_RETIREMENT = 21 | INACTIVE


# explicitly cast python Int to the status array type
INACTIVE_FLAG = np.uint8(Status.INACTIVE)
_INITIALISING = np.uint8(Status.INITIALISING)


@overload
def is_inactive(status: np.uint8) -> np.bool_: ...
@overload
def is_inactive(status: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]: ...
def is_inactive(status: np.uint8 | npt.NDArray[np.uint8]) -> np.bool_ | npt.NDArray[np.bool_]:
    """Check if particles are inactive.

    Parameters
    ----------
    status : np.uint8 or npt.NDArray[np.uint8]
        Status code of a single particle, or array of particle status codes.

    Returns
    -------
    np.bool_ or npt.NDArray[np.bool_]
        Whether the particle is inactive, or boolean array indicating inactive particles.
    """
    return (status & INACTIVE_FLAG) == INACTIVE_FLAG


@overload
def is_active(status: np.uint8) -> np.bool_: ...
@overload
def is_active(status: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]: ...
def is_active(status: np.uint8 | npt.NDArray[np.uint8]) -> np.bool_ | npt.NDArray[np.bool_]:
    """Check if particles are active.

    Parameters
    ----------
    status : np.uint8 or npt.NDArray[np.uint8]
        Status code of a single particle, or array of particle status codes.

    Returns
    -------
    np.bool_ or npt.NDArray[np.bool_]
        Whether the particle is active, or boolean array indicating active particles.
    """
    return np.logical_not(is_inactive(status))


def construct_initialise_status_kernel(status: Status) -> BoundKernel:
    """Construct a kernel that finalizes initialisation by setting a target status.

    Transitions particles with status ``Status.INITIALISING`` to the given `status`. Run
    automatically, last, by every :class:`~offline_particles.timestepping.Timestepper` at the end
    of its initialisation phase — see :meth:`Timestepper.run_initialisation`. Since
    initialisation runs every step (not just once at simulation start), this also finalizes
    particles that transition to ``Status.INITIALISING`` mid-simulation, e.g. via
    :func:`~offline_particles.kernels.timed_activation.construct_activate_released_particles_kernel`.

    Parameters
    ----------
    status : Status
        The status to transition ``Status.INITIALISING`` particles to.

    Returns
    -------
    BoundKernel
        A bound kernel that finalizes initialising particles to `status`.
    """
    target = np.uint8(status)

    @kernel_function(["status"])
    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def initialise_status(status_arr: npt.NDArray[np.uint8]) -> None:
        for i in numba.prange(status_arr.size):  # ty: ignore[not-iterable]
            if status_arr[i] == _INITIALISING:
                status_arr[i] = target

    kernel = ParticleKernel(initialise_status, particle_properties=[STATUS_DECLARATION])
    return BoundKernel(kernel)
