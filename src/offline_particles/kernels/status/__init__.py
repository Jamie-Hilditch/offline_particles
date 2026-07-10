import enum

import numpy as np
import numpy.typing as npt

__all__ = [
    "INACTIVE_FLAG",
    "Status",
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

    # timed releases and retirements
    PRE_RELEASE = 20 | INACTIVE
    POST_RETIREMENT = 21 | INACTIVE


# explicitly cast python Int to the status array type
INACTIVE_FLAG = np.uint8(Status.INACTIVE)


def is_inactive(status: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
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
    return (status & INACTIVE_FLAG) == INACTIVE_FLAG


def is_active(status: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
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
    return np.logical_not(is_inactive(status))
