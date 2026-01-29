"""Cython kernel functions for computing buoyancy forces on particles."""

from cython.parallel cimport prange

from .._core.inputs cimport unpack_field_data_3d
from .._core.interpolation.linear cimport trilinear_interpolation
from ..status cimport STATUS

cdef void _buoyancy_force_accumulation(particle_properties, scalars, field_data):
    """Add buoyancy force to particles assuming Boussinesq approximation."""
    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx
    cdef double[::1] rho_particle, rhs
    status = particle_properties["status"]
    zidx = particle_properties["zidx"]
    yidx = particle_properties["yidx"]
    xidx = particle_properties["xidx"]
    rho_particle = particle_properties["rho"]  # particle density
    rhs = particle_properties["rhs"]  # rhs to add buoyancy force to

    # unpack scalars
    cdef double rho0 = scalars["rho0"]
    cdef double g = scalars["g"]

    # unpack 3D field data
    cdef double[:, :, ::1] rho_array
    cdef double rho_offz, rho_offy, rho_offx
    rho_array, rho_offz, rho_offy, rho_offx = unpack_field_data_3d(field_data["rho"])

    # loop variables
    cdef double rho_environment

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # compute density at particle position
        rho_environment = trilinear_interpolation(
            rho_array,
            zidx[i] + rho_offz,
            yidx[i] + rho_offy,
            xidx[i] + rho_offx
        )

        # add to rhs
        rhs[i] += g * (rho_environment - rho_particle[i]) / rho0

# python wrappers
cpdef buoyancy_force_accumulation(particle_properties, scalars, field_data):
    _buoyancy_force_accumulation(particle_properties, scalars, field_data)
