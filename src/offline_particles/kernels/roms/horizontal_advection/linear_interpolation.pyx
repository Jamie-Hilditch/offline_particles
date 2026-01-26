"""Computing horizontal advection tendencies using linear interpolation."""

from cython.parallel cimport prange

from ..._core.inputs cimport unpack_field_data_2d, unpack_field_data_3d
from ..._core.interpolation.linear cimport bilinear_interpolation, trilinear_interpolation
from ...status cimport STATUS

cdef void _horizontal_idx_tendency_from_velocity_field(particle_properties, field_data):
    """Directly compute horizontal index tendency from velocity and grid spacing fields."""

    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx, didx
    status = particle_properties["status"]
    zidx = particle_properties["zidx"]
    yidx = particle_properties["yidx"]
    xidx = particle_properties["xidx"]
    didx = particle_properties["didx"]  # horizontal index tendency

    # no scalars needed

    # unpack 3D field data
    cdef double[:, :, ::1] vel_array
    cdef double vel_offz, vel_offy, vel_offx
    vel_array, vel_offz, vel_offy, vel_offx = unpack_field_data_3d(field_data["velocity"])

    # unpack 2D field data
    cdef double[:, ::1] grid_spacing_array
    cdef double grid_spacing_offy, grid_spacing_offx
    grid_spacing_array, grid_spacing_offy, grid_spacing_offx = unpack_field_data_2d(field_data["grid_spacing"])

    # loop variables
    cdef double vel_value, grid_spacing_value

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # interpolate velocity at particle position
        vel_value = trilinear_interpolation(
            vel_array,
            zidx[i] + vel_offz,
            yidx[i] + vel_offy,
            xidx[i] + vel_offx
        )
        # interpolate grid spacing at particle position
        grid_spacing_value = bilinear_interpolation(
            grid_spacing_array,
            yidx[i] + grid_spacing_offy,
            xidx[i] + grid_spacing_offx
        )
        # add to horizontal index tendency
        didx[i] += vel_value / grid_spacing_value

cdef void _horizontal_idx_tendency_from_velocity_property(particle_properties, field_data):
    """Compute horizontal index tendency from velocity particle property and grid spacings field."""

    # unpack required particle properties
    cdef unsigned char[::1] status
    cdef double[::1] yidx, xidx, didx, vel
    status = particle_properties["status"]
    yidx = particle_properties["yidx"]
    xidx = particle_properties["xidx"]
    didx = particle_properties["didx"]  # horizontal index tendency
    vel = particle_properties["vel"]  # particle velocity

    # no scalars needed

    # unpack 2D field data
    cdef double[:, ::1] grid_spacing_array
    cdef double grid_spacing_offy, grid_spacing_offx
    grid_spacing_array, grid_spacing_offy, grid_spacing_offx = unpack_field_data_2d(field_data["grid_spacing"])

    # loop variables
    cdef double grid_spacing_value

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # interpolate grid spacing at particle position
        grid_spacing_value = bilinear_interpolation(
            grid_spacing_array,
            yidx[i] + grid_spacing_offy,
            xidx[i] + grid_spacing_offx
        )

        # add to horizontal index tendency
        didx[i] += vel[i] / grid_spacing_value

# python wrappers

cpdef horizontal_idx_tendency_from_velocity_field(particle_properties, scalars, field_data):
    """Python wrapper for _horizontal_idx_tendency_from_velocity_field."""
    _horizontal_idx_tendency_from_velocity_field(particle_properties, field_data)

cpdef horizontal_idx_tendency_from_velocity_property(particle_properties, scalars, field_data):
    """Python wrapper for _horizontal_idx_tendency_from_velocity_property."""
    _horizontal_idx_tendency_from_velocity_property(particle_properties, field_data)
