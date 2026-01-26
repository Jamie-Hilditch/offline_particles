from cython.parallel cimport prange

from .._core cimport unpack_FieldData_3d
from .._interpolation.linear cimport trilinear_interpolation
from ..status cimport STATUS

import functools

import numpy as np

from .._kernels import ParticleKernel

# advection in physical space by linearly interpolating the vertical velocity w
cdef void _z_tendency_linearly_interpolate_w(particles, scalars, FieldData, dz_dt):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx
    cdef double[::1] dz
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx
    dz = particles[dz_dt]

    # unpack 3D field data
    cdef double[:, :, ::1] w_array
    cdef double w_offz, w_offy, w_offx
    w_array, w_offz, w_offy, w_offx = unpack_FieldData_3d(FieldData["w"])

    # loop variables
    cdef double w_value

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # compute vertical velocity at particle position
        w_value = trilinear_interpolation(
            w_array,
            zidx[i] + w_offz,
            yidx[i] + w_offy,
            xidx[i] + w_offx
        )

        # update vertical tendency
        dz[i] += w_value

# buoyancy-driven vertical advection tendency
cdef void _z_tendency_buoyancy_driven(particles, scalars, FieldData, dz_dt, dwb_dt):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx
    cdef double[::1] rho_p, wb, dz,  dwb
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx
    wb = particles.wb  # buoyancy velocity
    rho_p = particles.rho_p  # particle density
    dz = particles[dz_dt]
    dwb = particles[dwb_dt]

    # unpack scalars
    cdef double rho0 = scalars["rho0"]
    cdef double g = scalars["g"]
    cdef double damp = scalars["buoyancy_velocity_damping"]

    # unpack 3D field data
    cdef double[:, :, ::1] rho_array
    cdef double rho_offz, rho_offy, rho_offx
    rho_array, rho_offz, rho_offy, rho_offx = unpack_FieldData_3d(FieldData["rho"])

    # loop variables
    cdef double rho_value, buoyancy

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(nparticles, schedule='static', nogil=True):
        if status[i] & STATUS.INACTIVE:
            continue

        # compute density at particle position
        rho_value = trilinear_interpolation(
            rho_array,
            zidx[i] + rho_offz,
            yidx[i] + rho_offy,
            xidx[i] + rho_offx
        )

        # compute buoyancy
        buoyancy = g * (rho_value - rho_p[i]) / rho0

        # update buoyancy velocity tendency
        dwb[i] += buoyancy - damp * wb[i]

        # update vertical position tendency
        dz[i] += wb[i]

# python wrappers

cpdef z_tendency_linearly_interpolate_w(particles, scalars, FieldData, dz_dt):
    """
    Compute vertical advection tendency by linearly interpolating the
    vertical velocity field `w` at the particle positions.

    Parameters
    ----------
    particles : Particles
        The set of particles to operate on. Must contain the following fields:
        - status (uint8): particle status flags
        - zidx (double): vertical index position
        - yidx (double): eta index position
        - xidx (double): xi index position
        - [dz_dt] (double): z tendency to be updated


    scalars : dict[str, np.generic]
        A dictionary of scalar parameters required for the computation:
        - hc (double): critical depth parameter from ROMS.
        - NZ (int): number of vertical levels in the ROMS grid.

    FieldData : dict[str, FieldData]
        - w (double): vertical velocity field.

    dx_dt : str
        The name of the particle field to add the vertical tendency to (e.g., "_dz0").

    """
    _z_tendency_linearly_interpolate_w(particles, scalars, FieldData, dz_dt)

cpdef z_tendency_buoyancy_driven(particles, scalars, FieldData, dz_dt, dwb_dt):
    """
    Compute vertical advection tendency driven by buoyancy effects.

    Parameters
    ----------
    particles : ParticleSet
        The set of particles to operate on. Must contain the following fields:
        - status (uint8): particle status flags
        - zidx (double): vertical index position
        - yidx (double): eta index position
        - xidx (double): xi index position
        - rho_p (double): particle density
        - wb (double): buoyancy velocity
        - [dz_dt] (double): z tendency to be updated
        - [dwb_dt] (double): buoyancy velocity tendency to be updated

    scalars : dict[str, np.generic]
        A dictionary of scalar parameters required for the computation:
        - rho0 (double): reference density of the fluid.
        - g (double): acceleration due to gravity.
        - buoyancy_velocity_damping (double): damping coefficient for buoyancy velocity.

    FieldData : dict[str, FieldData]
        - rho (double): density field.

    dz_dt : str
        The name of the particle field to add the vertical tendency to (e.g., "_dz0").
    dwb_dt : str
        The name of the particle field to add the buoyancy velocity tendency to (e.g., "_dwb0").

    Notes
    -----
    This function updates the `dz_dt` and `dwb_dt` fields of each active particle with the
    effects of buoyancy on vertical motion. The buoyancy velocity is an additional vertical
    velocity added to `dz/dt` to account for differences between the particle density `rho_p`
    and fluid density `rho`. The buoyancy velocity is evolved according to
    `dwb/dt = g * (rho - rho_p) / rho0 - damp * wb` and the vertical position tendency is
    updated as `dz/dt += wb`.
    """
    _z_tendency_buoyancy_driven(particles, scalars, FieldData, dz_dt, dwb_dt)

# kernels


def z_tendency_linearly_interpolate_w_kernel(dz_dt: str) -> ParticleKernel:
    """
    Create a kernel to compute vertical advection tendency by linearly interpolating
    the vertical velocity field `w` at the particle positions.

    Parameters
    ----------
    dz_dt : str
        The name of the particle field to add the vertical tendency to (e.g., "_dz0").

    Returns
    -------
    ParticleKernel
        The vertical advection tendency kernel.
    """
    kernel_func = functools.partial(
        _z_tendency_linearly_interpolate_w,
        dz_dt=dz_dt
    )

    return ParticleKernel(
        kernel_func,
        particle_fields={
            "status": np.uint8,
            "zidx": np.float64,
            "yidx": np.float64,
            "xidx": np.float64,
            dz_dt: np.float64,
        },
        scalars={},
        simulation_fields=[
            "w"
        ],
    )


def z_tendency_buoyancy_driven_kernel(dz_dt: str, dwb_dt: str) -> ParticleKernel:
    """
    Create a kernel to compute buoyancy-driven vertical advection tendency.
    Parameters
    ----------
    dz_dt : str
        The name of the particle field to add the vertical tendency to (e.g., "_dz0").
    dwb_dt : str
        The name of the particle field to add the buoyancy velocity tendency to (e.g.,
        "_dwb0").
    Returns
    -------
    ParticleKernel
        The buoyancy-driven vertical advection tendency kernel.
    """
    kernel_func = functools.partial(
        _z_tendency_buoyancy_driven,
        dz_dt=dz_dt,
        dwb_dt=dwb_dt
    )
    return ParticleKernel(
        kernel_func,
        particle_fields={
            "status": np.uint8,
            "zidx": np.float64,
            "yidx": np.float64,
            "xidx": np.float64,
            "rho_p": np.float64,
            "wb": np.float64,
            dz_dt: np.float64,
            dwb_dt: np.float64,
        },
        scalars={
            "rho0": np.float64,
            "g": np.float64,
            "buoyancy_velocity_damping": np.float64,
        },
        simulation_fields=[
            "rho"
        ],
    )
