import numpy as np

from ...fields import FieldData
from ...particles import Particles
from .._kernels import ParticleKernel

def z_tendency_linearly_interpolate_w(
    particles: Particles, scalars: dict[str, np.generic], FieldData: dict[str, FieldData], dz_dt: str
) -> None: ...
def z_tendency_buoyancy_driven(
    particles: Particles, scalars: dict[str, np.generic], FieldData: dict[str, FieldData], dz_dt: str, dwb_dt: str
) -> None: ...
def z_tendency_linearly_interpolate_w_kernel(dz_dt: str) -> ParticleKernel: ...
def z_tendency_buoyancy_driven_kernel(dz_dt: str, dwb_dt: str) -> ParticleKernel: ...
