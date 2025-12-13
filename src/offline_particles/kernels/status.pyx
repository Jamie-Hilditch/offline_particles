"""Kernels that check the status of particles."""

from cython.parallel cimport prange
from libc.math cimport isfinite

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
    idx = particles.yidx
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

    cdef double zmin = scalars["zmin"]
    cdef double zmax = scalars["zmax"]
    cdef double ymin = scalars["ymin"]
    cdef double ymax = scalars["ymax"]
    cdef double xmin = scalars["xmin"]
    cdef double xmax = scalars["xmax"]

    cdef Py_ssize_t i, n
    n = status.shape[0]

    for i in prange(n, schedule="static", nogil=True):
        # Skip inactive particles
        if status[i] != 0:
            continue

        # if any index is out of bounds mark as invalid
        if (zidx[i] < 0.0 or zidx[i] > zmax or
            yidx[i] < 0.0 or yidx[i] > ymax or
            xidx[i] < 0.0 or xidx[i] > xmax):
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
        "zmin": np.float64,
        "zmax": np.float64,
        "ymin": np.float64,
        "ymax": np.float64,
        "xmin": np.float64,
        "xmax": np.float64,
    },
    simulation_fields=[],
)

validation_kernel = ParticleKernel.chain(finite_indices_kernel, domain_bounds_kernel)