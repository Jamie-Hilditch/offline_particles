"""Define some common kernel inputs."""

import numpy as np
import numpy.typing as npt

from ._kernels import ParticlePropertyDeclaration, ScalarDeclaration

# particle properties
STATUS_DECLARATION = ParticlePropertyDeclaration("status", np.uint8)
ZIDX_DECLARATION = ParticlePropertyDeclaration("zidx", np.float64)
YIDX_DECLARATION = ParticlePropertyDeclaration("yidx", np.float64)
XIDX_DECLARATION = ParticlePropertyDeclaration("xidx", np.float64)

# scalars
DT_DECLARATION = ScalarDeclaration("_dt", np.float64)


def construct_time_declaration(time_dtype: npt.DTypeLike) -> ScalarDeclaration:
    """Construct a ScalarDeclaration for the simulation time.

    Args:
        time_dtype: Data type of the simulation time. Use np.float64 for float-based clocks.
            For datetime-based clocks, pass an explicit datetime64 dtype with a unit that matches
            the simulation clock's time array, e.g. np.dtype('datetime64[ns]').
    """
    return ScalarDeclaration("_time", np.dtype(time_dtype))
