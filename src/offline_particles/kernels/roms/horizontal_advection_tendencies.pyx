"""Kernels for computing horizontal advection in ROMS models."""

from cython.parallel cimport prange

from .._core cimport unpack_FieldData_2d, unpack_FieldData_3d
from .._interpolation.linear cimport trilinear_interpolation, bilinear_interpolation
from ..status cimport STATUS

import functools

import numpy as np

from .._kernels import ParticleKernel

# compute the horizontal tendencies in index space using linear interpolation

cdef void _xyidx_tendency_linear_interpolation(particles, scalars, FieldData, dxidx_dt, dyidx_dt):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx
    cdef double[::1] dxidx, dyidx
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx
    dxidx = particles[dxidx_dt]
    dyidx = particles[dyidx_dt]

    # no scalars needed

    # unpack 3D field data
    cdef double[:, :, ::1] u_array, v_array
    cdef double u_offz, u_offy, u_offx
    cdef double v_offz, v_offy, v_offx
    u_array, u_offz, u_offy, u_offx = unpack_FieldData_3d(FieldData["u"])
    v_array, v_offz, v_offy, v_offx = unpack_FieldData_3d(FieldData["v"])

    # unpack 2D field data
    cdef double[:, ::1] dx_array, dy_array
    cdef double dx_offy, dx_offx
    cdef double dy_offy, dy_offx
    dx_array, dx_offy, dx_offx = unpack_FieldData_2d(FieldData["dx"])
    dy_array, dy_offy, dy_offx = unpack_FieldData_2d(FieldData["dy"])

    # loop variables
    cdef double u_value, v_value, dx_value, dy_value

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # first compute the derivative values for the current particle positions
        # first dxidx0
        u_value = trilinear_interpolation(
            u_array,
            zidx[i] + u_offz,
            yidx[i] + u_offy,
            xidx[i] + u_offx
        )
        dx_value = bilinear_interpolation(
            dx_array,
            yidx[i] + dx_offy,
            xidx[i] + dx_offx
        )
        dxidx[i] += u_value / dx_value

        # next dyidx0
        v_value = trilinear_interpolation(
            v_array,
            zidx[i] + v_offz,
            yidx[i] + v_offy,
            xidx[i] + v_offx
        )
        dy_value = bilinear_interpolation(
            dy_array,
            yidx[i] + dy_offy,
            xidx[i] + dy_offx
        )
        dyidx[i] += v_value / dy_value

cpdef xyidx_tendency_linear_interpolation(particles, scalars, FieldData, dxidx_dt, dyidx_dt):
    """Compute horizontal advection tendencies in index space using linear interpolation.

    This kernel computes the horizontal advection tendencies for particles in index space
    using linear interpolation of the velocity fields.

    Parameters
    ----------
    particles : Particles
        The particles to compute tendencies for. Must have the following fields:
        - status (uint8): particle status flags
        - zidx (double): vertical index position
        - yidx (double): eta index position
        - xidx (double): xi index position
        - [dxidx_dt] (double): tendency in xi index space (to be updated)
        - [dyidx_dt] (double): tendency in eta index space (to be updated)
    scalars : dict[str, np.generic]
        None required by this kernel.
    FieldData : dict[str, FieldData]
        The field data containing:
        - u (double): 3D xi velocity field
        - v (double): 3D eta velocity field
        - dx (double): 2D grid spacing in the xi direction
        - dy (double): 2D grid spacing in the eta direction
    dxidx_dt : str
        The name of the particle field to add xi index space tendency to.
    dyidx_dt : str
        The name of the particle field to add eta index space tendency to.
    """
    _xyidx_tendency_linear_interpolation(particles, scalars, FieldData, dxidx_dt, dyidx_dt)

# kernel


def xyidx_tendency_linear_interpolation_kernel(dxidx_dt: str, dyidx_dt: str) -> ParticleKernel:
    """
    Create a kernel to compute horizontal advection tendencies in index space
    using linear interpolation.

    Parameters
    ----------
    dxidx_dt : str
        The name of the particle field to add xi index space tendency to.
    dyidx_dt : str
        The name of the particle field to add eta index space tendency to.

    Returns
    -------
    ParticleKernel
        The horizontal advection tendency kernel.
    """
    kernel_func = functools.partial(
        _xyidx_tendency_linear_interpolation,
        dxidx_dt=dxidx_dt,
        dyidx_dt=dyidx_dt
    )

    return ParticleKernel(
        kernel_func,
        particle_fields={
            "status": np.uint8,
            "zidx": np.float64,
            "yidx": np.float64,
            "xidx": np.float64,
            dxidx_dt: np.float64,
            dyidx_dt: np.float64,
        },
        scalars={},
        simulation_fields=[
            "u", "v", "dx", "dy"
        ]
    )
