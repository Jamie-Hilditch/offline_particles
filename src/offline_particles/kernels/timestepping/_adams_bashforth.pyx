"""Kernel functions for implementing Adams-Bashforth schemes."""

from cython.parallel cimport prange

from .._core cimport float_t
from ..status cimport STATUS

import numpy as np

cdef void _ab2_update_impl(
    unsigned char[::1] status,
    float_t[::1] prop,
    float_t[::1] dprop_0,
    float_t[::1] dprop_1,
    float_t dt
) noexcept nogil:

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static'):
        if status[i] & STATUS.INACTIVE:
            continue

        # handle initialization steps
        if status[i] == STATUS.MULTISTEP_1:
            # if on first step use forward Euler, i.e. set prior step derivatives equal to current
            dprop_1[i] = dprop_0[i]

        # update field using AB2 scheme
        prop[i] += dt * (dprop_0[i] * 1.5 - dprop_1[i] * 0.5)

        # shift derivatives for next time step
        dprop_1[i] = dprop_0[i]
        dprop_0[i] = 0.0  # reset current tendency for next accumulation

cdef _ab2_update_float(particle_properties, float dt):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef float [::1] prop, dprop_0, dprop_1
    status = particle_properties["status"]
    prop = particle_properties["prop"]
    dprop_0 = particle_properties["dprop_0"]
    dprop_1 = particle_properties["dprop_1"]

    _ab2_update_impl(status, prop, dprop_0, dprop_1, dt)

cdef _ab2_update_double(particle_properties, double dt):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] prop, dprop_0, dprop_1
    status = particle_properties["status"]
    prop = particle_properties["prop"]
    dprop_0 = particle_properties["dprop_0"]
    dprop_1 = particle_properties["dprop_1"]

    _ab2_update_impl(status, prop, dprop_0, dprop_1, dt)

cdef void _ab2_bump_status(particle_properties):
    # unpack required particle fields
    cdef unsigned char[::1] status
    status = particle_properties["status"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # update status to indicate multistep has been initialized
        if status[i] == STATUS.MULTISTEP_1:
            status[i] = STATUS.NORMAL

cdef void _ab2_initialisation(particle_properties):
    # unpack required particle fields
    cdef unsigned char[::1] status
    status = particle_properties["status"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        status[i] == STATUS.MULTISTEP_1

cdef void _ab3_update_impl(
    unsigned char[::1] status,
    float_t[::1] prop,
    float_t[::1] dprop_0,
    float_t[::1] dprop_1,
    float_t[::1] dprop_2,
    float_t dt
) noexcept nogil:
    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static'):
        if status[i] & STATUS.INACTIVE:
            continue

        # handle initialization steps
        if status[i] == STATUS.MULTISTEP_1:
            # if on first step use forward Euler, i.e. set prior step derivatives equal to current
            dprop_2[i] = dprop_1[i]
            dprop_1[i] = dprop_0[i]
        elif status[i] == STATUS.MULTISTEP_2:
            # if on second step set df2 to be consistent with AB2
            dprop_2[i] = 2.0 * dprop_1[i] - dprop_0[i]

        # update field using AB3 scheme
        prop[i] += dt * (dprop_0[i] * 23.0 - dprop_1[i] * 16.0 + dprop_2[i] * 5.0) / 12.0

        # shift derivatives for next time step
        dprop_2[i] = dprop_1[i]
        dprop_1[i] = dprop_0[i]
        dprop_0[i] = 0.0  # reset current tendency for next accumulation

cdef _ab3_update_float(particle_properties, float dt):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef float [::1] prop, dprop_0, dprop_1, dprop_2
    status = particle_properties["status"]
    prop = particle_properties["prop"]
    dprop_0 = particle_properties["dprop_0"]
    dprop_1 = particle_properties["dprop_1"]
    dprop_2 = particle_properties["dprop_2"]

    _ab3_update_impl(status, prop, dprop_0, dprop_1, dprop_2, dt)

cdef _ab3_update_double(particle_properties, double dt):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] prop, dprop_0, dprop_1, dprop_2
    status = particle_properties["status"]
    prop = particle_properties["prop"]
    dprop_0 = particle_properties["dprop_0"]
    dprop_1 = particle_properties["dprop_1"]
    dprop_2 = particle_properties["dprop_2"]

    _ab3_update_impl(status, prop, dprop_0, dprop_1, dprop_2, dt)

cdef _ab3_bump_status(particle_properties):
    # unpack required particle fields
    cdef unsigned char[::1] status
    status = particle_properties["status"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # update status to indicate multistep has been initialized
        if status[i] == STATUS.MULTISTEP_1:
            status[i] = STATUS.MULTISTEP_2
        elif status[i] == STATUS.MULTISTEP_2:
            status[i] = STATUS.NORMAL

cdef _ab3_initialisation(particle_properties):
    # unpack required particle fields
    cdef unsigned char[::1] status
    status = particle_properties["status"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        status[i] == STATUS.MULTISTEP_2

# python wrappers
cpdef ab2_update_float32(particle_properties, scalars, field_data):
    """
    Update float32 particle property using 2nd-order Adams-Bashforth scheme.
    """
    _ab2_update_float(particle_properties, np.float32(scalars["_dt"]))

cpdef ab2_update_float64(particle_properties, scalars, field_data):
    """
    Update float64 particle property using 2nd-order Adams-Bashforth scheme.
    """
    _ab2_update_double(particle_properties, scalars["_dt"])

cpdef ab2_bump_status(particle_properties, scalars, field_data):
    """
    Bump particle status after Adams-Bashforth 2nd-order update.
    """
    _ab2_bump_status(particle_properties)

cpdef ab2_initialisation(particle_properties, scalars, field_data):
    """
    Initialise particle status for Adams-Bashforth 2nd-order scheme.
    """
    _ab2_initialisation(particle_properties)

cpdef ab3_update_float32(particle_properties, scalars, field_data):
    """
    Update float32 particle property using 3rd-order Adams-Bashforth scheme.
    """
    _ab3_update_float(particle_properties, np.float32(scalars["_dt"]))

cpdef ab3_update_float64(particle_properties, scalars, field_data):
    """
    Update float64 particle property using 3rd-order Adams-Bashforth scheme.
    """
    _ab3_update_double(particle_properties, scalars["_dt"])

cpdef ab3_bump_status(particle_properties, scalars, field_data):
    """
    Bump particle status after Adams-Bashforth 3rd-order update.
    """
    _ab3_bump_status(particle_properties)

cpdef ab3_initialisation(particle_properties, scalars, field_data):
    """
    Initialise particle status for Adams-Bashforth 3rd-order scheme.
    """
    _ab3_initialisation(particle_properties)
