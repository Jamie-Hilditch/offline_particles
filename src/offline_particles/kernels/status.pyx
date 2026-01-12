"""Particle status codes."""

from .status cimport STATUS

from enum import IntEnum


class ParticleStatus(IntEnum):
    # inactive flag
    INACTIVE = STATUS.INACTIVE

    # normal state
    NORMAL = STATUS.NORMAL

    # error states
    NONFINITE = STATUS.NONFINITE
    OUT_OF_DOMAIN = STATUS.OUT_OF_DOMAIN
    BELOW_BOTTOM = STATUS.BELOW_BOTTOM
    ABOVE_SURFACE = STATUS.ABOVE_SURFACE

    # Reserved for multistep initialization
    MULTISTEP_1 = STATUS.MULTISTEP_1
    MULTISTEP_2 = STATUS.MULTISTEP_2


def is_inactive(status: npt.NDArray[np.unit8]) -> npt.NDArray[np.bool_]:
    """Check if particles are inactive.

    Parameters
    ----------
    status : npt.NDArray[np.unit8]
        Array of particle status codes.

    Returns
    -------
    npt.NDArray[np.bool_]
        Boolean array indicating active particles.
    """
    return (status & ParticleStatus.INACTIVE) == ParticleStatus.INACTIVE


def is_active(status: npt.NDArray[np.unit8]) -> npt.NDArray[np.bool_]:
    """Check if particles are active.

    Parameters
    ----------
    status : npt.NDArray[np.unit8]
        Array of particle status codes.

    Returns
    -------
    npt.NDArray[np.bool_]
        Boolean array indicating active particles.
    """
    return ~is_inactive(status)
