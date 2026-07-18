"""Define some common kernel inputs."""

import numpy as np
import numpy.typing as npt

from ._kernels import ParticlePropertyDeclaration, ScalarDeclaration

__all__ = [
    "DT_DECLARATION",
    "STATUS_DECLARATION",
    "XIDX_DECLARATION",
    "YIDX_DECLARATION",
    "ZIDX_DECLARATION",
    "construct_time_declaration",
]

# particle properties
#: The :class:`ParticlePropertyDeclaration` for the particle status.
STATUS_DECLARATION = ParticlePropertyDeclaration(
    "status", np.uint8, description="The particle status. Possible values are defined by the `Status` enum."
)
#: The :class:`ParticlePropertyDeclaration` for the particle's index along the z-axis.
ZIDX_DECLARATION = ParticlePropertyDeclaration(
    "zidx",
    np.float64,
    description="The fractional index of the particle along the z-axis with respect to the centred grid.",
)
#: The :class:`ParticlePropertyDeclaration` for the particle's index along the y-axis.
YIDX_DECLARATION = ParticlePropertyDeclaration(
    "yidx",
    np.float64,
    description="The fractional index of the particle along the y-axis with respect to the centred grid.",
)
#: The :class:`ParticlePropertyDeclaration` for the particle's index along the x-axis.
XIDX_DECLARATION = ParticlePropertyDeclaration(
    "xidx",
    np.float64,
    description="The fractional index of the particle along the x-axis with respect to the centred grid.",
)

# scalars
#: The :class:`ScalarDeclaration` for the (non-dimensional) simulation time step.
DT_DECLARATION = ScalarDeclaration("_dt", np.float64, description="The current (non-dimensional) simulation time step.")


def construct_time_declaration(time_dtype: npt.DTypeLike) -> ScalarDeclaration:
    """Construct a ScalarDeclaration for the simulation time.

    Parameters
    ----------
    time_dtype : npt.DTypeLike
        Data type of the simulation time. Use np.float64 for float-based clocks.
        For datetime-based clocks, pass an explicit datetime64 dtype with a unit that matches
        the simulation clock's time array, e.g. np.dtype('datetime64[ns]').

    Returns
    -------
    ScalarDeclaration
        A ScalarDeclaration for the simulation time.
    """
    return ScalarDeclaration("_time", np.dtype(time_dtype).type, description="The current simulation time.")
