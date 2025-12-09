"""Simple interpolation kernels for computing output for example."""

import numpy as np
import numpy.typing as npt

from ..interpolation import bilinear_interpolation, trilinear_interpolation
from ..kernel_tools import offset_indices_2D, offset_indices_3D
from ..particle_kernel import ParticleKernel


def create_trilinear_interpolation_kernel(field: str, particle_field: str | None = None, dtype: npt.DTypeLike = np.float64) ->ParticleKernel:
    """Create a ParticleKernel that performs trilinear interpolation."""
    if particle_field is None:
        particle_field = field

    def kernel_fn(
        particles,
        pidx,
        field_data,
        field_off,
    ):
        idx = offset_indices_3D(particles["zidx"][pidx], particles["yidx"][pidx], particles["xidx"][pidx], field_off)
        interp_value = trilinear_interpolation(idx, field_data)
        particles[particle_field][pidx] = interp_value

    return ParticleKernel(
        kernel_fn,
        particle_fields={"zidx": float, "yidx": float, "xidx": float, particle_field: dtype},
        scalars=(),
        simulation_fields=[field],
    )

def create_horizontal_bilinear_interpolation_kernel(field: str, particle_field: str | None = None, dtype: npt.DTypeLike = np.float64) ->ParticleKernel:
    """Create a ParticleKernel that performs bilinear interpolation."""
    if particle_field is None:
        particle_field = field

    def kernel_fn(
        particles,
        pidx,
        field_data,
        field_off,
    ):
        idx = offset_indices_2D(particles["yidx"][pidx], particles["xidx"][pidx], field_off)
        interp_value = bilinear_interpolation(idx, field_data)
        particles[particle_field][pidx] = interp_value

    return ParticleKernel(
        kernel_fn,
        particle_fields={"yidx": float, "xidx": float, particle_field: dtype},
        scalars=(),
        simulation_fields=[field],
    )