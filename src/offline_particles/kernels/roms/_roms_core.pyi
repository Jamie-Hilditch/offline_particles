import numpy as np

from ...fields import FieldData
from ...particles import Particles
from .._kernels import ParticleKernel

def compute_z_kernel_function(
    particles: Particles, scalars: dict[str, np.number], fielddata: dict[str, FieldData], particle_field: str
) -> None: ...
def compute_z_kernel(particle_field: str = "z") -> ParticleKernel: ...
