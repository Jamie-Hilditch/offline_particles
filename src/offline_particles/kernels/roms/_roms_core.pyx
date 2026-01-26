"""Some core ROMS functions implemented in Cython."""

from cython.parallel cimport prange

from .._core cimport unpack_FieldData_1d, unpack_FieldData_2d
from .._interpolation.linear cimport bilinear_interpolation, linear_interpolation
from ..status cimport STATUS
from ._vertical_coordinate cimport compute_z, compute_zidx


import functools

import numpy as np

from .._kernels import ParticleKernel

cdef void _compute_z(particles, scalars, FieldData, particle_field):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx, z
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx
    z = particles[particle_field]

    # unpack scalars
    cdef double hc = scalars["hc"]
    cdef int NZ = scalars["NZ"]

    # unpack 2D field data
    cdef double[:, ::1] h_array, zeta_array
    cdef double h_offy, h_offx
    cdef double zeta_offy, zeta_offx
    h_array, h_offy, h_offx = unpack_FieldData_2d(FieldData["h"])
    zeta_array, zeta_offy, zeta_offx = unpack_FieldData_2d(FieldData["zeta"])

    # unpack 1D field data
    cdef double[::1] C_array
    cdef double C_offz
    C_array, C_offz = unpack_FieldData_1d(FieldData["C"])

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

cdef _compute_zidx(particles, scalars, FieldData, particle_field):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx, z
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx
    z = particles[particle_field]

    # unpack scalars
    cdef double hc = scalars["hc"]
    cdef int NZ = scalars["NZ"]

    # unpack 2D field data
    cdef double[:, ::1] h_array, zeta_array
    cdef double h_offy, h_offx
    cdef double zeta_offy, zeta_offx
    h_array, h_offy, h_offx = unpack_FieldData_2d(FieldData["h"])
    zeta_array, zeta_offy, zeta_offx = unpack_FieldData_2d(FieldData["zeta"])

    # unpack 1D field data
    cdef double[::1] C_array
    cdef double C_offz
    C_array, C_offz = unpack_FieldData_1d(FieldData["C"])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    # declare loop variables
    cdef double h_value, zeta_value

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

cpdef compute_z_kernel_function(particles, scalars, FieldData, particle_field):
    """Compute the physical vertical coordinate for particles."""
    return _compute_z(particles, scalars, FieldData, particle_field)

cpdef compute_zidx_kernel_function(particles, scalars, FieldData, particle_field):
    """Compute the vertical index for particles."""
    return _compute_zidx(particles, scalars, FieldData, particle_field)


def compute_z_kernel(particle_field: str = "z") -> ParticleKernel:
    """Return a kernel that computes the physical vertical coordinate for particles.

    Parameters
    ----------
    particle_field : str
        The name of the particle field to store the computed physical vertical coordinate.

    Returns
    -------
    ParticleKernel
        A particle kernel that computes the physical vertical coordinate.
    """
    kernel_func = functools.partial(_compute_z, particle_field=particle_field)
    return ParticleKernel(
        kernel_func,
        particle_fields={
            "status": np.uint8,
            "zidx": np.float64,
            "yidx": np.float64,
            "xidx": np.float64,
            particle_field: np.float64
        },
        scalars={
            "hc": np.float64,
            "NZ": np.int32
        },
        simulation_fields=[
            "h",
            "C",
            "zeta",
        ],
    )


def compute_zidx_kernel(particle_field: str = "z") -> ParticleKernel:
    """Return a kernel that computes the vertical index for particles.
    Parameters
    ----------
    particle_field : str
        The name of the particle field that contains the physical vertical coordinate.
    Returns
    -------
    ParticleKernel
        A particle kernel that computes the vertical index.
    """
    kernel_func = functools.partial(_compute_zidx, particle_field=particle_field)
    return ParticleKernel(
        kernel_func,
        particle_fields={
            "status": np.uint8,
            "zidx": np.float64,
            "yidx": np.float64,
            "xidx": np.float64,
            particle_field: np.float64
        },
        scalars={
            "hc": np.float64,
            "NZ": np.int32
        },
        simulation_fields=[
            "h",
            "C",
            "zeta",
        ],
    )
