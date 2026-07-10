import enum
from typing import overload

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
