import numpy as np

from ...fields import FieldData
from ...particles import Particles
from .._kernels import ParticleKernel

def xyidx_tendency_linear_interpolation(
    particles: Particles, scalars: dict[str, np.generic], FieldData: dict[str, FieldData], dxidx_dt: str, dyidx_dt: str
): ...
def xyidx_tendency_linear_interpolation_kernel(dxidx_dt: str, dyidx_dt: str) -> ParticleKernel: ...
