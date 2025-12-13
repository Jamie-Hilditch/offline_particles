"""Advection scheme froms ROMS data with vertical velocity w.

Horizontal advection is done in index space. However, ROMS uses a sigma
coordinate system in the vertical, so vertical advection requires special treatment.
This module implements vertical advection in physical space using the vertical
velocity w and then transforms that into index space.
"""

from cython.parallel cimport prange

from .._core cimport unpack_fielddata_1d, unpack_fielddata_2d, unpack_fielddata_3d
from .._interpolation.linear cimport trilinear_interpolation, bilinear_interpolation, linear_interpolation
from ._vertical_coordinate cimport compute_z, compute_zidx

import functools

from ._kernels import ParticleKernel
from ...timesteppers import RK2Timestepper


cdef void _rk2_step_1(particles, scalars, fielddata):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx, z, dxidx1, dyidx1, dz1
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx
    z = particles.z
    dxidx1 = particles._dxidx1
    dyidx1 = particles._dyidx1
    dz1 = particles._dz1


    # unpack scalars 
    cdef double hc = scalars["hc"]
    cdef int NZ = scalars["NZ"]

    # unpack 3D field data
    cdef double[:, :, ::1] u_array, v_array, w_array
    cdef double u_offz, u_offy, u_offx
    cdef double v_offz, v_offy, v_offx
    cdef double w_offz, w_offy, w_offx
    u_array, u_offz, u_offy, u_offx = unpack_fielddata_3d(fielddata["u"])
    v_array, v_offz, v_offy, v_offx = unpack_fielddata_3d(fielddata["v"])
    w_array, w_offz, w_offy, w_offx = unpack_fielddata_3d(fielddata["w"])

    
    # unpack 2D field data
    cdef double[:, ::1] dx_array, dy_array, h_array, zeta_array
    cdef double dx_offy, dx_offx
    cdef double dy_offy, dy_offx
    cdef double h_offy, h_offx
    cdef double zeta_offy, zeta_offx
    dx_array, dx_offy, dx_offx = unpack_fielddata_2d(fielddata["dx"])
    dy_array, dy_offy, dy_offx = unpack_fielddata_2d(fielddata["dy"])
    h_array, h_offy, h_offx = unpack_fielddata_2d(fielddata["h"])
    zeta_array, zeta_offy, zeta_offx = unpack_fielddata_2d(fielddata["zeta"])

    # unpack 1D field data
    cdef double[::1] C_array
    cdef double C_offz
    C_array, C_offz = unpack_fielddata_1d(fielddata["C"])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    # declare loop variables
    cdef double h_value, zeta_value, C_value
    cdef double u_value, v_value, dx_value, dy_value

    for i in prange(nparticles, schedule='static', nogil=True):

        # skip inactive particles
        if status[i] != 0:
            continue 

        # first compute z 
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

        # then interpolate horizontal velocities in index space
        u_value = trilinear_interpolation(
            u_array,    
            zidx[i] + u_offz,
            yidx[i] + u_offy,
            xidx[i] + u_offx
        )
        v_value = trilinear_interpolation(
            v_array,
            zidx[i] + v_offz,
            yidx[i] + v_offy,
            xidx[i] + v_offx
        )
        dx_value = bilinear_interpolation(
            dx_array,
            yidx[i] + dx_offy,
            xidx[i] + dx_offx
        )
        dy_value = bilinear_interpolation(
            dy_array,
            yidx[i] + dy_offy,
            xidx[i] + dy_offx
        )
        dxidx1[i] = u_value / dx_value
        dyidx1[i] = v_value / dy_value

        # finally do vertical advection in physical space
        dz1[i] = trilinear_interpolation(
            w_array,
            zidx[i] + w_offz,
            yidx[i] + w_offy,
            xidx[i] + w_offx
        )

           

cdef void _rk2_step_2(particles, scalars, fielddata):
    # unpack required particle fields
    cdef unsigned char[::1] status 
    cdef double[::1] zidx, yidx, xidx, z, dxidx1, dyidx1, dz1, dxidx2, dyidx2, dz2
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx
    z = particles.z
    dxidx1 = particles._dxidx1
    dyidx1 = particles._dyidx1
    dz1 = particles._dz1
    dxidx2 = particles._dxidx2
    dyidx2 = particles._dyidx2
    dz2 = particles._dz2

    # unpack scalars 
    cdef double dt = scalars["_dt"]
    cdef double alpha = scalars["_RK2_alpha"]
    cdef double hc = scalars["hc"]
    cdef int NZ = scalars["NZ"]

     # unpack 3D field data
    cdef double[:, :, ::1] u_array, v_array, w_array
    cdef double u_offz, u_offy, u_offx
    cdef double v_offz, v_offy, v_offx
    cdef double w_offz, w_offy, w_offx
    u_array, u_offz, u_offy, u_offx = unpack_fielddata_3d(fielddata["u"])
    v_array, v_offz, v_offy, v_offx = unpack_fielddata_3d(fielddata["v"])
    w_array, w_offz, w_offy, w_offx = unpack_fielddata_3d(fielddata["w"])
    
    # unpack 2D field data
    cdef double[:, ::1] dx_array, dy_array, h_array, zeta_array
    cdef double dx_offy, dx_offx
    cdef double dy_offy, dy_offx
    cdef double h_offy, h_offx
    cdef double zeta_offy, zeta_offx
    dx_array, dx_offy, dx_offx = unpack_fielddata_2d(fielddata["dx"])
    dy_array, dy_offy, dy_offx = unpack_fielddata_2d(fielddata["dy"])
    h_array, h_offy, h_offx = unpack_fielddata_2d(fielddata["h"])
    zeta_array, zeta_offy, zeta_offx = unpack_fielddata_2d(fielddata["zeta"])


    # unpack 1D field data
    cdef double[::1] C_array
    cdef double C_offz
    C_array, C_offz = unpack_fielddata_1d(fielddata["C"])

    # loop over particles
    cdef Py_ssize_t i, nparticles 
    nparticles = status.shape[0]

    # declare loop variables
    cdef double z_int, yidx_int, xidx_int
    cdef double h_value, zeta_value, C_value, zidx_int
    cdef double u_value, v_value, dx_value, dy_value


    for i in prange(nparticles, schedule='static', nogil=True):
        # skip inactive particles
        if status[i] != 0:
            continue 

        # intermediate positions
        z_int = z[i] + alpha * dt * dz1[i]
        yidx_int = yidx[i] + alpha * dt * dyidx1[i]
        xidx_int = xidx[i] + alpha * dt * dxidx1[i]

        # compute intermediate zidx
        h_value = bilinear_interpolation(
            h_array, 
            yidx_int + h_offy, 
            xidx_int + h_offx
        )
        zeta_value = bilinear_interpolation(
            zeta_array, 
            yidx_int + zeta_offy, 
            xidx_int + zeta_offx
        )
        zidx_int = compute_zidx(z_int, h_value, zeta_value, hc, NZ, C_array, C_offz)

        # interpolate horizontal velocities in index space
        u_value = trilinear_interpolation(
            u_array,    
            zidx_int + u_offz,
            yidx_int + u_offy,
            xidx_int + u_offx
        )
        v_value = trilinear_interpolation(
            v_array,
            zidx_int + v_offz,
            yidx_int + v_offy,
            xidx_int + v_offx
        )
        dx_value = bilinear_interpolation(
            dx_array,
            yidx_int + dx_offy,
            xidx_int + dx_offx
        )
        dy_value = bilinear_interpolation(
            dy_array,
            yidx_int + dy_offy,
            xidx_int + dy_offx
        )
        dxidx2[i] = u_value / dx_value
        dyidx2[i] = v_value / dy_value

        # finally do vertical advection in physical space
        dz2[i] = trilinear_interpolation(
            w_array,
            zidx_int + w_offz,
            yidx_int + w_offy,
            xidx_int + w_offx
        )
            

cdef void _rk2_update(particles, scalars, fielddata):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx, z, dxidx1, dyidx1, dz1, dxidx2, dyidx2, dz2
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx
    z = particles.z
    dxidx1 = particles._dxidx1
    dyidx1 = particles._dyidx1
    dz1 = particles._dz1
    dxidx2 = particles._dxidx2
    dyidx2 = particles._dyidx2
    dz2 = particles._dz2

    # unpack scalars 
    cdef double dt = scalars["_dt"]
    cdef double alpha = scalars["_RK2_alpha"]
    cdef double hc = scalars["hc"]
    cdef int NZ = scalars["NZ"]
    
    # unpack 2D field data
    cdef double[:, ::1] h_array, zeta_array
    cdef double h_offy, h_offx
    cdef double zeta_offy, zeta_offx
    h_array, h_offy, h_offx = unpack_fielddata_2d(fielddata["h"])
    zeta_array, zeta_offy, zeta_offx = unpack_fielddata_2d(fielddata["zeta"])

    # unpack 1D field data
    cdef double[::1] C_array
    cdef double C_offz
    C_array, C_offz = unpack_fielddata_1d(fielddata["C"])

    # loop over particles
    cdef Py_ssize_t i, nparticles 
    nparticles = status.shape[0]

    # rk constants
    cdef double b1 = 1.0 / (2.0 * alpha)
    cdef double b2 = 1.0 - b1

    # declare loop variables
    cdef double h_value, zeta_value

    for i in prange(nparticles, schedule='static', nogil=True):
        # skip inactive particles
        if status[i] != 0:
            continue 

        # update positions
        z[i] = z[i] + b1 * dt * dz1[i] + b2 * dt * dz2[i]
        yidx[i] = yidx[i] + b1 * dt * dyidx1[i] + b2 * dt * dyidx2[i]
        xidx[i] = xidx[i] + b1 * dt * dxidx1[i] + b2 * dt * dxidx2[i]

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

# define python wrappers
cpdef rk2_w_advection_step_1(particles, scalars, fielddata):
    """First step of RK2 with ROMS w advection."""
    _rk2_step_1(particles, scalars, fielddata)

cpdef rk2_w_advection_step_2(particles, scalars, fielddata):
    """Second step of RK2 with ROMS w advection."""
    _rk2_step_2(particles, scalars, fielddata)

cpdef rk2_w_advection_update(particles, scalars, fielddata):
    """Update step of RK2 with ROMS w advection."""
    _rk2_update(particles, scalars, fielddata)

# define kernels
rk2_w_advection_step_1_kernel = ParticleKernel(
    rk2_w_advection_step_1,
    particle_fields={
        "status": np.uint8,
        "zidx": np.float64, 
        "yidx": np.float64, 
        "xidx": np.float64, 
        "z": np.float64, 
        "_dxidx1": np.float64, 
        "_dyidx1": np.float64, 
        "_dz1": np.float64
    },
    scalars={
        "hc": np.float64, 
        "NZ": np.int32
    },
    simulation_fields=[
        "u",
        "v",
        "w",
        "dx",
        "dy",
        "h",
        "C",
        "zeta",
    ],
)
rk2_w_advection_step_2_kernel = ParticleKernel(
    rk2_w_advection_step_2,
    particle_fields={
        "status": np.uint8,
        "zidx": np.float64, 
        "yidx": np.float64, 
        "xidx": np.float64, 
        "z": np.float64, 
        "_dxidx1": np.float64, 
        "_dyidx1": np.float64, 
        "_dz1": np.float64,
        "_dxidx2": np.float64, 
        "_dyidx2": np.float64, 
        "_dz2": np.float64
    },
    scalars={
        "_dt": np.float64,
        "_RK2_alpha": np.float64,
        "hc": np.float64, 
        "NZ": np.int32
    },
    simulation_fields=[
        "u",
        "v",
        "w",
        "dx",
        "dy",
        "h",
        "C",
        "zeta",
    ],
)
rk2_w_advection_update_kernel = ParticleKernel(
    rk2_w_advection_update,
    particle_fields={
        "status": np.uint8,
        "zidx": np.float64, 
        "yidx": np.float64, 
        "xidx": np.float64, 
        "z": np.float64, 
        "_dxidx1": np.float64, 
        "_dyidx1": np.float64, 
        "_dz1": np.float64,
        "_dxidx2": np.float64, 
        "_dyidx2": np.float64, 
        "_dz2": np.float64
    },
    scalars={
        "_dt": np.float64,
        "_RK2_alpha": np.float64,
        "hc": np.float64, 
        "NZ": np.int32
    },
    simulation_fields=[
        "h",
        "C",
        "zeta",
    ],
)

# define time stepper
"""Create an RK2 timesteppers with the ROMS w advection kernels."""
rk2_w_advection_timestepper = functools.partial(
    RK2Timestepper,
    rk_step_1_kernel=rk2_w_advection_step_1_kernel,
    rk_step_2_kernel=rk2_w_advection_step_2_kernel,
    rk_update_kernel=rk2_w_advection_update_kernel,
)