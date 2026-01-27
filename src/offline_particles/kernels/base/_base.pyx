"""Cython kernel functions implementing basic particle operations."""

from cython.parallel cimport prange

from .._core cimport prop_t, float_t
from ...status cimport STATUS


cdef void _copy_property(particle_properties):
    """Copy a particle property from source to destination."""

    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef prop_t[::1] source, destination
    status = particle_properties["status"]
    source = particle_properties["source"]
    destination = particle_properties["destination"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # copy source to destination
        destination[i] = source[i]

cdef void _add_property(particle_properties):
    """Add a particle property from source to destination."""

    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef prop_t[::1] source, destination
    status = particle_properties["status"]
    source = particle_properties["source"]
    destination = particle_properties["destination"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # add source to destination
        destination[i] += source[i]

cdef _subtract_property(particle_properties):
    """Subtract a particle property source from destination."""

    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef prop_t[::1] source, destination
    status = particle_properties["status"]
    source = particle_properties["source"]
    destination = particle_properties["destination"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # subtract source from destination
        destination[i] -= source[i]

cdef _multiply_property(particle_properties):
    """Multiply a particle property destination by source."""

    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef prop_t[::1] source, destination
    status = particle_properties["status"]
    source = particle_properties["source"]
    destination = particle_properties["destination"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # multiply destination by source
        destination[i] *= source[i]

cdef _divide_property(particle_properties):
    """Divide a particle property destination by source.

    Note this is only defined for float types.
    """

    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef float_t[::1] source, destination
    status = particle_properties["status"]
    source = particle_properties["source"]
    destination = particle_properties["destination"]

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # divide destination by source
        destination[i] /= source[i]

# python wrappers

cpdef copy_property(particle_properties, scalars, field_data):
    _copy_property(particle_properties)

cpdef add_property(particle_properties, scalars, field_data):
    _add_property(particle_properties)

cpdef subtract_property(particle_properties, scalars, field_data):
    _subtract_property(particle_properties)

cpdef multiply_property(particle_properties, scalars, field_data):
    _multiply_property(particle_properties)

cpdef divide_property(particle_properties, scalars, field_data):
    _divide_property(particle_properties)
