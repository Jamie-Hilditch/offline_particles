"""Cython kernel functions for applying damping and relaxation to particle properties."""

from cython.parallel cimport prange
from libc.math cimport fabs

from ...status cimport STATUS

cdef void _linear_damping_accumulation(particle_properties, scalars):
    """Apply linear damping to particle property."""
    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef double[::1] prop, rhs
    status = particle_properties["status"]
    prop = particle_properties["prop"]
    rhs = particle_properties["rhs"]

    # unpack scalars
    cdef double drag_coeff = scalars["linear_damping_coefficient"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # apply linear drag
        rhs[i] -= drag_coeff * prop[i]

cdef void _quadratic_damping_accumulation(particle_properties, scalars):
    """Apply quadratic drag to particle property."""
    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef double[::1] prop, rhs
    status = particle_properties["status"]
    prop = particle_properties["prop"]
    rhs = particle_properties["rhs"]

    # unpack scalars
    cdef double drag_coeff = scalars["quadratic_damping_coefficient"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # apply quadratic drag
        rhs[i] -= drag_coeff * prop[i] * fabs(prop[i])

cdef void _linear_relaxation_accumulation(particle_properties, scalars):
    """Apply linear relaxation of particle property to a target value."""
    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef double[::1] prop, target, rhs
    status = particle_properties["status"]
    prop = particle_properties["prop"]
    target = particle_properties["target"]
    rhs = particle_properties["rhs"]

    # unpack scalars
    cdef double rc = scalars["linear_relaxation_coefficient"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # apply linear relaxation to field value
        rhs[i] += rc * (target[i] - prop[i])

cdef void _quadratic_relaxation_accumulation(particle_properties, scalars):
    """Apply quadratic relaxation of particle property to a target value."""
    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef double[::1] prop, target, rhs
    status = particle_properties["status"]
    prop = particle_properties["prop"]
    target = particle_properties["target"]
    rhs = particle_properties["rhs"]

    # unpack scalars
    cdef double rc = scalars["quadratic_relaxation_coefficient"]

    # loop variables
    cdef double delta

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # apply quadratic relaxation to field value
        delta = target[i] - prop[i]
        rhs[i] += rc * (delta * fabs(delta))

# python wrappers

cpdef linear_damping_accumulation(particle_properties, scalars, field_data):
    _linear_damping_accumulation(particle_properties, scalars)

cpdef quadratic_damping_accumulation(particle_properties, scalars, field_data):
    _quadratic_damping_accumulation(particle_properties, scalars)

cpdef linear_relaxation_accumulation(particle_properties, scalars, field_data):
    _linear_relaxation_accumulation(particle_properties, scalars)

cpdef quadratic_relaxation_accumulation(particle_properties, scalars, field_data):
    _quadratic_relaxation_accumulation(particle_properties, scalars)
