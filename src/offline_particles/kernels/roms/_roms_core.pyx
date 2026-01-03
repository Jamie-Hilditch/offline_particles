"""Some core ROMS functions implemented in Cython."""

from cython.parallel cimport prange

from .._core cimport unpack_fielddata_1d, unpack_fielddata_2d
from .._interpolation.linear cimport bilinear_interpolation, linear_interpolation
from ..status cimport STATUS
from ._vertical_coordinate cimport compute_z


import functools

import numpy as np

from .._kernels import ParticleKernel

cdef void _compute_z(particles, scalars, fielddata, particle_field):
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
    h_array, h_offy, h_offx = unpack_fielddata_2d(fielddata["h"])
    zeta_array, zeta_offy, zeta_offx = unpack_fielddata_2d(fielddata["zeta"])

    # unpack 1D field data
    cdef double[::1] C_array
    cdef double C_offz
    C_array, C_offz = unpack_fielddata_1d(fielddata["C"])

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

cpdef compute_z_kernel_function(particles, scalars, fielddata, particle_field):
    """Compute the physical vertical coordinate for particles."""
    return _compute_z(particles, scalars, fielddata, particle_field)


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
