"""Define some common kernel inputs."""

import numpy as np

from ._kernels import ParticlePropertyDeclaration, ScalarDeclaration

# particle properties
STATUS_DECLARATION = ParticlePropertyDeclaration("status", np.uint8)
ZIDX_DECLARATION = ParticlePropertyDeclaration("zidx", np.float64)
YIDX_DECLARATION = ParticlePropertyDeclaration("yidx", np.float64)
XIDX_DECLARATION = ParticlePropertyDeclaration("xidx", np.float64)

# scalars
DT_DECLARATION = ScalarDeclaration("_dt", np.float64)


def construct_time_declaration(time_dtype: type[np.float64] | type[np.datetime64]) -> ScalarDeclaration:
    """Construct a ScalarDeclaration for the simulation time.

    Args:
        time_dtype: Data type of the simulation time. Supported types are np.float64 and np.datetime64.
    """
    return ScalarDeclaration("_time", np.dtype(time_dtype))


# field data
