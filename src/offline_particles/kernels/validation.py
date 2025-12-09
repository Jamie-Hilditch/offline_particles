"""Kernels for validating particle status."""

import numpy as np
import numpy.typing as npt

from ..particle_kernel import ParticleKernel


def _finite_indices(
    particles: npt.NDArray,
    pidx: int
) -> None:
    """Kernel to check if particle positions are finite."""
    if (
        not np.isfinite(particles["zidx"][pidx])
        or not np.isfinite(particles["yidx"][pidx])
        or not np.isfinite(particles["xidx"][pidx])
    ):
        particles["status"][pidx] = 1 # Mark particle as inactive if any position is not finite
        
finite_indices_kernel = ParticleKernel(
    _finite_indices,
    particle_fields={
        "status": np.uint8,
        "zidx": np.float64,
        "yidx": np.float64,
        "xidx": np.float64,
    },
    scalars=(),
    simulation_fields=[],
    fastmath=False
)

def _inbounds(
    particles: npt.NDArray,
    pidx: int,
    zidx_min: float,
    zidx_max: float,
    yidx_min: float,
    yidx_max: float,
    xidx_min: float,
    xidx_max: float,
) -> None:
    """Kernel to check if particle indices are in bounds."""
    if (
        particles["zidx"][pidx] < zidx_min
        or particles["zidx"][pidx] > zidx_max
        or particles["yidx"][pidx] < yidx_min
        or particles["yidx"][pidx] > yidx_max
        or particles["xidx"][pidx] < xidx_min
        or particles["xidx"][pidx] > xidx_max
    ):
        particles["status"][pidx] = 2 # Mark particle as inactive if any index is out of bounds
        
inbounds_kernel = ParticleKernel(
    _inbounds,
    particle_fields={
        "status": np.uint8,
        "zidx": np.float64,
        "yidx": np.float64,
        "xidx": np.float64,
    },
    scalars=("zidx_min", "zidx_max", "yidx_min", "yidx_max", "xidx_min", "xidx_max"),
    simulation_fields=[],
)

validate_indices_kernel = ParticleKernel.from_sequence(
    [finite_indices_kernel, inbounds_kernel])