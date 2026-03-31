import enum

import numpy as np
import numpy.typing as npt

from ._status import STATUS_VALUES

__all__ = [
    "INACTIVE_FLAG",
    "Status",
    "is_active",
    "is_inactive",
]


class Status(enum.IntEnum):
    """Enumeration of particle status codes."""

    INACTIVE = STATUS_VALUES["INACTIVE"]
    NORMAL = STATUS_VALUES["NORMAL"]
    NONFINITE = STATUS_VALUES["NONFINITE"]
    OUT_OF_DOMAIN = STATUS_VALUES["OUT_OF_DOMAIN"]
    BELOW_BOTTOM = STATUS_VALUES["BELOW_BOTTOM"]
    ABOVE_SURFACE = STATUS_VALUES["ABOVE_SURFACE"]
    MULTISTEP_1 = STATUS_VALUES["MULTISTEP_1"]
    MULTISTEP_2 = STATUS_VALUES["MULTISTEP_2"]
    PRE_RELEASE = STATUS_VALUES["PRE_RELEASE"]
    POST_RETIREMENT = STATUS_VALUES["POST_RETIREMENT"]


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
