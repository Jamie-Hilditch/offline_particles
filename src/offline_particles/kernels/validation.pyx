"""Kernels that check the status of particles."""

from cython.parallel cimport prange
from libc.math cimport isfinite

import numpy as np

from ._kernels import ParticleKernel

# expose python objects as public API
__all__ = [
    "finite_indices",
    "domain_bounds",
    "finite_indices_kernel",
    "domain_bounds_kernel",
    "validation_kernel",
]

cdef void _finite_indices(particles):
    """
    Sets particles.status[i] = 1 if any indices are not finite.
    """
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx

    # loop over particles
    cdef Py_ssize_t i, n
    n = status.shape[0]

    for i in prange(n, schedule="static", nogil=True):
        # Skip inactive particles
        if status[i] != 0:
            continue

        # if any index is non-finite mark as invalid
        if not isfinite(zidx[i]) or not isfinite(yidx[i]) or not isfinite(xidx[i]):
            status[i] = 1

cdef _domain_bounds(particles, scalars, fielddata):
    """
    Sets particles.status[i] = 2 if any indices are out of bounds.
    """
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx

    cdef double zmin = scalars["zidx_min"]
    cdef double zmax = scalars["zidx_max"]
    cdef double ymin = scalars["yidx_min"]
    cdef double ymax = scalars["yidx_max"]
    cdef double xmin = scalars["xidx_min"]
    cdef double xmax = scalars["xidx_max"]

    cdef Py_ssize_t i, n
    n = status.shape[0]

    for i in prange(n, schedule="static", nogil=True):
        # Skip inactive particles
        if status[i] != 0:
            continue

        # if any index is out of bounds mark as invalid
        if not (
            zmin <= zidx[i] <= zmax and
            ymin <= yidx[i] <= ymax and
            xmin <= xidx[i] <= xmax
        ):
            status[i] = 2


# Python wrapper functions
cpdef finite_indices(particles, scalars, fielddata):
    """
    Check particle indices are finite.
    """
    _finite_indices(particles)

cpdef domain_bounds(particles, scalars, fielddata):
    """
    Check particle indices are within domain bounds.
    """
    _domain_bounds(particles, scalars, fielddata)

# Define ParticleKernel instances
finite_indices_kernel = ParticleKernel(
    finite_indices,
    particle_fields={
        "status": np.uint8,
        "zidx": np.float64,
        "yidx": np.float64,
        "xidx": np.float64,
    },
    scalars={},
    simulation_fields=[],
)
domain_bounds_kernel = ParticleKernel(
    domain_bounds,
    particle_fields={
        "status": np.uint8,
        "zidx": np.float64,
        "yidx": np.float64,
        "xidx": np.float64,
    },
    scalars={
        "zidx_min": np.float64,
        "zidx_max": np.float64,
        "yidx_min": np.float64,
        "yidx_max": np.float64,
        "xidx_min": np.float64,
        "xidx_max": np.float64,
    },
    simulation_fields=[],
)

validation_kernel = ParticleKernel.chain(finite_indices_kernel, domain_bounds_kernel)
