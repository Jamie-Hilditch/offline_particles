"""Advection scheme froms ROMS data with vertical velocity w.

Horizontal advection is done in index space. However, ROMS uses a sigma
coordinate system in the vertical, so vertical advection requires special treatment.
This module implements vertical advection in physical space using the vertical
velocity w and then transforms that into index space.
"""

from cython.parallel cimport prange

from .._core cimport unpack_fielddata_1d, unpack_fielddata_2d, unpack_fielddata_3d
from .._interpolation.linear cimport trilinear_interpolation, bilinear_interpolation, linear_interpolation
from ..status cimport STATUS
from ._vertical_coordinate cimport compute_z, compute_zidx

import numpy as np

from .._kernels import ParticleKernel

cdef void _ab3_advection(particles, scalars, fielddata):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx, z
    cdef double[::1] dz0, dz1, dz2
    cdef double[::1] dyidx0, dyidx1, dyidx2
    cdef double[::1] dxidx0, dxidx1, dxidx2
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx
    z = particles.z
    dz0 = particles._dz0
    dz1 = particles._dz1
    dz2 = particles._dz2
    dyidx0 = particles._dyidx0
    dyidx1 = particles._dyidx1
    dyidx2 = particles._dyidx2
    dxidx0 = particles._dxidx0
    dxidx1 = particles._dxidx1
    dxidx2 = particles._dxidx2

    # unpack scalars
    cdef double dt = scalars["_dt"]
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

    # loop variables
    cdef double h_value, zeta_value, C_value
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
        dxidx0[i] = u_value / dx_value

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
        dyidx0[i] = v_value / dy_value

        # finally dz0
        dz0[i] = trilinear_interpolation(
            w_array,
            zidx[i] + w_offz,
            yidx[i] + w_offy,
            xidx[i] + w_offx
        )

        # handle initialization steps
        if status[i] == STATUS.MULTISTEP_1:
            # if on first step we must fill in both prior steps
            # use forward Euler, i.e. set prior step derivatives equal to current
            dxidx1[i] = dxidx0[i]
            dyidx1[i] = dyidx0[i]
            dz1[i] = dz0[i]
            dxidx2[i] = dxidx0[i]
            dyidx2[i] = dyidx0[i]
            dz2[i] = dz0[i]
            status[i] = STATUS.MULTISTEP_2
        elif status[i] == STATUS.MULTISTEP_2:
            # if on second step fill in derivatives from n-2
            # use dy2 = -dy0 + 2*dy1 to give AB2 consistency
            dxidx2[i] = -dxidx0[i] + 2.0 * dxidx1[i]
            dyidx2[i] = -dyidx0[i] + 2.0 * dyidx1[i]
            dz2[i] = -dz0[i] + 2.0 * dz1[i]
            status[i] = STATUS.NORMAL

        # compute physical depth at current position
        # In practice this is already set from prior step, but we
        # recompute here working on the principle that only
        # zidx, yidx, xidx are guaranteed to be correct at the start of the step
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

        # step forward z, yidx, xidx using AB3
        z[i] += dt * (23.0 * dz0[i] - 16.0 * dz1[i] + 5.0 * dz2[i]) / 12.0
        yidx[i] += dt * (23.0 * dyidx0[i] - 16.0 * dyidx1[i] + 5.0 * dyidx2[i]) / 12.0
        xidx[i] += dt * (23.0 * dxidx0[i] - 16.0 * dxidx1[i] + 5.0 * dxidx2[i]) / 12.0

        # shift derivative histories
        dz2[i] = dz1[i]
        dxidx2[i] = dxidx1[i]
        dyidx2[i] = dyidx1[i]
        dz1[i] = dz0[i]
        dxidx1[i] = dxidx0[i]
        dyidx1[i] = dyidx0[i]

cdef void _ab3_post_advection(particles, scalars, fielddata):
    """Post-advection step to update particle zidx from z."""
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx, z
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx
    z = particles.z

    # unpack scalars
    cdef double hc = scalars["hc"]
    cdef int NZ = scalars["NZ"]

    # upack 2D field data
    cdef double[:, ::1] h_array, zeta_array
    cdef double h_offy, h_offx
    cdef double zeta_offy, zeta_offx
    h_array, h_offy, h_offx = unpack_fielddata_2d(fielddata["h"])
    zeta_array, zeta_offy, zeta_offx = unpack_fielddata_2d(fielddata["zeta"])

    # unpack 1D field data
    cdef double[::1] C_array
    cdef double C_offz
    C_array, C_offz = unpack_fielddata_1d(fielddata["C"])

    # loop variables
    cdef double h_value, zeta_value

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # compute new zidx from updated physical depth
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


# python wrapper
cpdef ab3_advection(particles, scalars, fielddata):
    """Advect particles using 3rd-order Adams-Bashforth scheme using wy."""
    _ab3_advection(particles, scalars, fielddata)

cpdef ab3_post_advection(particles, scalars, fielddata):
    """Post-advection step to update particle zidx from z."""
    _ab3_post_advection(particles, scalars, fielddata)

# kernel
ab3_w_advection_kernel = ParticleKernel(
    ab3_advection,
    particle_fields={
        "status": np.uint8,
        "zidx": np.float64,
        "yidx": np.float64,
        "xidx": np.float64,
        "_dz0": np.float64,
        "_dz1": np.float64,
        "_dz2": np.float64,
        "_dyidx0": np.float64,
        "_dyidx1": np.float64,
        "_dyidx2": np.float64,
        "_dxidx0": np.float64,
        "_dxidx1": np.float64,
        "_dxidx2": np.float64,
    },
    scalars={
        "_dt": np.float64,
        "hc": np.float64,
        "NZ": np.int32,
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

ab3_post_advection_kernel = ParticleKernel(
    ab3_post_advection,
    particle_fields={
        "status": np.uint8,
        "zidx": np.float64,
        "yidx": np.float64,
        "xidx": np.float64,
        "z": np.float64,
    },
    scalars={
        "hc": np.float64,
        "NZ": np.int32,
    },
    simulation_fields=[
        "h",
        "C",
        "zeta",
    ],
)
