import enum

import numpy as np
import numpy.typing as npt

from ._status import STATUS_VALUES

__all__ = [
    "Status",
    "is_active",
    "is_inactive",
]

Status = enum.IntEnum("Status", STATUS_VALUES)
Status.__doc__ = """Enumeration of particle status codes."""


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
    return (status & Status.INACTIVE) == Status.INACTIVE


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
    return np.logical_not(is_inactive(status))
