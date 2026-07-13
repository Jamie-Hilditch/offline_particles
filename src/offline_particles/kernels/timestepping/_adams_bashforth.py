"""Implementations of Adams-Bashforth time-stepping schemes."""

import numba
import numpy as np
import numpy.typing as npt

from ..status import INACTIVE_FLAG, Status

# constants for use in Adams-Bashforth kernels to check for multistep status
_NORMAL = np.uint8(Status.NORMAL)
_MULTISTEP_1 = np.uint8(Status.MULTISTEP_1)
_MULTISTEP_2 = np.uint8(Status.MULTISTEP_2)


@numba.njit(parallel=True, nogil=True, fastmath=True)
def ab2_update(
    status: npt.NDArray[np.uint8],
    prop: npt.NDArray[np.number],
    dprop_0: npt.NDArray[np.number],
    dprop_1: npt.NDArray[np.number],
    dt: np.float64,
) -> None:
    """Adams-Bashforth 2 update kernel.

    Args:
        status: Particle status array.
        prop: Particle property array to be updated.
        dprop_0: Particle property derivative at the current time step.
        dprop_1: Particle property derivative at the previous time step.
        dt: Time step size.
    """
    for i in numba.prange(status.size):  # ty: ignore[not-iterable]
        if status[i] & INACTIVE_FLAG:
            continue

        if status[i] == _MULTISTEP_1:
            # if on first step use forward Euler, i.e. set prior step derivatives equal to current
            dprop_1[i] = dprop_0[i]

        # update field using AB2 scheme
        prop[i] += dt * (dprop_0[i] * 1.5 - dprop_1[i] * 0.5)

        # shift derivatives for next time step
        dprop_1[i] = dprop_0[i]
        dprop_0[i] = 0.0  # reset current tendency for next accumulation


@numba.njit(parallel=True, nogil=True, fastmath=True)
def ab3_update(
    status: npt.NDArray[np.uint8],
    prop: npt.NDArray[np.number],
    dprop_0: npt.NDArray[np.number],
    dprop_1: npt.NDArray[np.number],
    dprop_2: npt.NDArray[np.number],
    dt: np.float64,
) -> None:
    """Adams-Bashforth 3 update kernel.

    Args:
        status: Particle status array.
        prop: Particle property array to be updated.
        dprop_0: Particle property derivative at the current time step.
        dprop_1: Particle property derivative at the previous time step.
        dprop_2: Particle property derivative at the time step before the previous one.
        dt: Time step size.
    """
    for i in numba.prange(status.size):  # ty: ignore[not-iterable]
        if status[i] & INACTIVE_FLAG:
            continue

        if status[i] == _MULTISTEP_1:
            # if on first step use forward Euler, i.e. set prior step derivatives equal to current
            dprop_1[i] = dprop_0[i]
            dprop_2[i] = dprop_0[i]
        elif status[i] == _MULTISTEP_2:
            # if on second step set dprop_2 to be consistent with AB2
            dprop_2[i] = 2.0 * dprop_1[i] - dprop_0[i]

        # update field using AB3 scheme
        prop[i] += dt * (dprop_0[i] * 23 / 12 - dprop_1[i] * 16 / 12 + dprop_2[i] * 5 / 12)

        # shift derivatives for next time step
        dprop_2[i] = dprop_1[i]
        dprop_1[i] = dprop_0[i]
        dprop_0[i] = 0.0  # reset current tendency for next accumulation


@numba.njit(parallel=True, nogil=True, fastmath=True)
def ab_bump_status(status: npt.NDArray[np.uint8]) -> None:
    """Adams-Bashforth bump status kernel to update multistep status flags after each time step.

    Args:
        status: Particle status array.
    """
    for i in numba.prange(status.size):  # ty: ignore[not-iterable]
        if status[i] == _MULTISTEP_1:
            # if on first step, set to normal for next step
            status[i] = _NORMAL
        elif status[i] == _MULTISTEP_2:
            # if on second step, set to multistep 1 for next step
            status[i] = _MULTISTEP_1
