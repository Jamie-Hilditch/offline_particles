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

# field data
