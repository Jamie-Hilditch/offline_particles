"""Cython extension module exposing kernel functions for ROMS vertical coordinate functions."""

from cython.parallel cimport prange

from ..._core.inputs cimport unpack_field_data_1d, unpack_field_data_2d
from ..._core.interpolation.linear cimport bilinear_interpolation, linear_interpolation
from ...status cimport STATUS
from ._vertical_coordinate cimport compute_z, compute_zidx

cdef void _compute_z_kernel_function(particle_properties, scalars, field_data):
    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx, z
    status = particle_properties["status"]
    zidx = particle_properties["zidx"]
    yidx = particle_properties["yidx"]
    xidx = particle_properties["xidx"]
    z = particle_properties["z"]

    # unpack scalars
    cdef double hc = scalars["hc"]
    cdef int NZ = scalars["NZ"]

    # unpack 2D field data
    cdef double[:, ::1] h_array, zeta_array
    cdef double h_offy, h_offx
    cdef double zeta_offy, zeta_offx
    h_array, h_offy, h_offx = unpack_field_data_2d(field_data["h"])
    zeta_array, zeta_offy, zeta_offx = unpack_field_data_2d(field_data["zeta"])

    # unpack 1D field data
    cdef double[::1] C_array
    cdef double C_offz
    C_array, C_offz = unpack_field_data_1d(field_data["C"])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):

        if status[i] & STATUS.INACTIVE:  # only compute for active particles
            continue

        h_value = bilinear_interpolation(
            h_array,
            yidx[i] + h_offy,
            xidx[i] + h_offx
        )
        zeta_value = bilinear_interpolation(
            zeta_array,
            yidx[i] + zeta_offy,
            xidx[i] + zeta_offx
        )
        C_value = linear_interpolation(
            C_array,
            zidx[i] + C_offz
        )
        z[i] = compute_z(
            zidx[i],
            NZ,
            hc,
            h_value,
            C_value,
            zeta_value
        )

cdef void _compute_zidx_kernel_function(particle_properties, scalars, field_data):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx, z
    status = particle_properties["status"]
    zidx = particle_properties["zidx"]
    yidx = particle_properties["yidx"]
    xidx = particle_properties["xidx"]
    z = particle_properties["z"]

    # unpack scalars
    cdef double hc = scalars["hc"]
    cdef int NZ = scalars["NZ"]

    # unpack 2D field data
    cdef double[:, ::1] h_array, zeta_array
    cdef double h_offy, h_offx
    cdef double zeta_offy, zeta_offx
    h_array, h_offy, h_offx = unpack_field_data_2d(field_data["h"])
    zeta_array, zeta_offy, zeta_offx = unpack_field_data_2d(field_data["zeta"])

    # unpack 1D field data
    cdef double[::1] C_array
    cdef double C_offz
    C_array, C_offz = unpack_field_data_1d(field_data["C"])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(0, nparticles, schedule='static', nogil=True):
        # skip inactive particles
        if status[i] & STATUS.INACTIVE:
            continue

        # compute zidx
        h_value = bilinear_interpolation(
            h_array,
            yidx[i] + h_offy,
            xidx[i] + h_offx
        )
        zeta_value = bilinear_interpolation(
            zeta_array,
            yidx[i] + zeta_offy,
            xidx[i] + zeta_offx
        )
        zidx[i] = compute_zidx(z[i], h_value, zeta_value, hc, NZ, C_array, C_offz)

# python wrappers

cpdef compute_z_kernel_function(particle_properties, scalars, field_data):
    """Compute the vertical position 'z' of particles in ROMS vertical coordinates.

    Parameters
    ----------
    particle_properties : dict
        Dictionary containing particle properties 'status', 'zidx', 'yidx', 'xidx', and 'z'.
    scalars : dict
        Dictionary containing scalar parameters 'hc' and 'NZ'.
    field_data : dict
        Dictionary containing field data 'h', 'zeta', and 'C'.
    """
    _compute_z_kernel_function(particle_properties, scalars, field_data)

cpdef compute_zidx_kernel_function(particle_properties, scalars, field_data):
    """Compute the vertical index 'zidx' of particles in ROMS vertical coordinates.

    Parameters
    ----------
    particle_properties : dict
        Dictionary containing particle properties 'status', 'zidx', 'yidx', 'xidx', and 'z'.
    scalars : dict
        Dictionary containing scalar parameters 'hc' and 'NZ'.
    field_data : dict
        Dictionary containing field data 'h', 'zeta', and 'C'.
    """
    _compute_zidx_kernel_function(particle_properties, scalars, field_data)
