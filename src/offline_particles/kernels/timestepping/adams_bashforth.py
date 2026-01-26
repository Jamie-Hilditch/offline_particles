"""ParticleKernels for Adams-Bashforth timestepping schemes."""

import numpy as np

from .._kernels import ParticleKernel, ParticlePropertyDeclaration
from ..common_inputs import DT_DECLARATION, STATUS_DECLARATION
from ._adams_bashforth import ab2_bump_status, ab2_update, ab3_bump_status, ab3_update

# particle property declarations for Adams-Bashforth
prop_declaration = ParticlePropertyDeclaration("prop", np.float64)
dprop_0_declaration = ParticlePropertyDeclaration("dprop_0", np.float64)
dprop_1_declaration = ParticlePropertyDeclaration("dprop_1", np.float64)

ab2_update_kernel = ParticleKernel(
    ab2_update,
    particle_properties=[
        STATUS_DECLARATION,
        prop_declaration,
        dprop_0_declaration,
    ],
    scalars=[DT_DECLARATION],
)
ab2_bump_status_kernel = ParticleKernel(
    ab2_bump_status,
    particle_properties=[STATUS_DECLARATION],
)
ab3_update_kernel = ParticleKernel(
    ab3_update,
    particle_properties=[
        STATUS_DECLARATION,
        prop_declaration,
        dprop_0_declaration,
        dprop_1_declaration,
    ],
    scalars=[DT_DECLARATION],
)
ab3_bump_status_kernel = ParticleKernel(
    ab3_bump_status,
    particle_properties=[STATUS_DECLARATION],
)
