"""Kernel functions that check the status of particles."""

from cython.parallel cimport prange
from libc.math cimport isfinite

from ..status cimport STATUS

cdef void _finite_indices(particle_properties):
    """
    Sets status[i] = STATUS.FINITE if any indices are not finite.
    """
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx
    status = particle_properties["status"]
    zidx = particle_properties["zidx"]
    yidx = particle_properties["yidx"]
    xidx = particle_properties["xidx"]

    # loop over particles
    cdef Py_ssize_t i, n
    n = status.shape[0]

    for i in prange(n, schedule="static", nogil=True):
        # Skip inactive particles
        if status[i] & STATUS.INACTIVE:
            continue

        # if any index is non-finite mark as invalid
        if not isfinite(zidx[i]) or not isfinite(yidx[i]) or not isfinite(xidx[i]):
            status[i] = STATUS.NONFINITE

cdef void _domain_bounds(particle_properties, scalars):
    """
    Sets status[i] = STATUS.OUT_OF_DOMAIN if either horizontal index is out of bounds.
    Sets status[i] = STATUS.BELOW_BOTTOM if vertical index is less than min.
    Sets status[i] = STATUS.ABOVE_SURFACE if vertical index is greater than max.
    """
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx
    status = particle_properties["status"]
    zidx = particle_properties["zidx"]
    yidx = particle_properties["yidx"]
    xidx = particle_properties["xidx"]

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
        if status[i] & STATUS.INACTIVE:
            continue

        # check vertical index
        if zidx[i] < zmin:
            status[i] = STATUS.BELOW_BOTTOM
        elif zidx[i] > zmax:
            status[i] = STATUS.ABOVE_SURFACE

        # if any index is out of bounds mark as invalid
        # note this takes precedence over vertical checks
        if not (
            ymin <= yidx[i] <= ymax and
            xmin <= xidx[i] <= xmax
        ):
            status[i] = STATUS.OUT_OF_DOMAIN

# Python wrapper functions
cpdef finite_indices(particle_properties, scalars, field_data):
    """
    Check particle indices are finite.
    """
    _finite_indices(particle_properties)

cpdef domain_bounds(particle_properties, scalars, field_data):
    """
    Check particle indices are within domain bounds.
    """
    _domain_bounds(particle_properties, scalars)
