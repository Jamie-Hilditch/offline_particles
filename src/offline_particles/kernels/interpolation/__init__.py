"""Interpolation kernels."""

import functools

from ._kernel_constructors import (
    construct_1D_interpolation_kernel,
    construct_2D_interpolation_kernel,
    construct_3D_interpolation_kernel,
    construct_X_interpolation_kernel,
    construct_XY_interpolation_kernel,
    construct_XYZ_interpolation_kernel,
    construct_XZ_interpolation_kernel,
    construct_XZY_interpolation_kernel,
    construct_Y_interpolation_kernel,
    construct_YX_interpolation_kernel,
    construct_YXZ_interpolation_kernel,
    construct_YZ_interpolation_kernel,
    construct_YZX_interpolation_kernel,
    construct_Z_interpolation_kernel,
    construct_ZX_interpolation_kernel,
    construct_ZXY_interpolation_kernel,
    construct_ZY_interpolation_kernel,
    construct_ZYX_interpolation_kernel,
)
from ._lagrange import (
    lagrange2N_1D_factory,
    lagrange2N_1D_particle_factory,
    lagrange2N_2D_factory,
    lagrange2N_2D_particle_factory,
    lagrange2N_3D_factory,
    lagrange2N_3D_particle_factory,
)

__all__ = [
    "lagrange2N_1D_factory",
    "lagrange2N_2D_factory",
    "lagrange2N_3D_factory",
    "lagrange2N_1D_particle_factory",
    "lagrange2N_2D_particle_factory",
    "lagrange2N_3D_particle_factory",
    "linear_interpolation_factory",
    "bilinear_interpolation_factory",
    "trilinear_interpolation_factory",
    "cubic_interpolation_factory",
    "bicubic_interpolation_factory",
    "tricubic_interpolation_factory",
    "linear_interpolation_particle",
    "bilinear_interpolation_particle",
    "trilinear_interpolation_particle",
    "cubic_interpolation_particle",
    "bicubic_interpolation_particle",
    "tricubic_interpolation_particle",
    "construct_1D_interpolation_kernel",
    "construct_2D_interpolation_kernel",
    "construct_3D_interpolation_kernel",
    "construct_Z_interpolation_kernel",
    "construct_Y_interpolation_kernel",
    "construct_X_interpolation_kernel",
    "construct_XY_interpolation_kernel",
    "construct_XZ_interpolation_kernel",
    "construct_YX_interpolation_kernel",
    "construct_YZ_interpolation_kernel",
    "construct_ZX_interpolation_kernel",
    "construct_ZY_interpolation_kernel",
    "construct_XYZ_interpolation_kernel",
    "construct_XZY_interpolation_kernel",
    "construct_YXZ_interpolation_kernel",
    "construct_YZX_interpolation_kernel",
    "construct_ZXY_interpolation_kernel",
    "construct_ZYX_interpolation_kernel",
]

# aliases and special cases of Lagrange interpolation functions
#: linear interpolation is a special case of 1D Lagrange interpolation with N=1
linear_interpolation_factory = functools.partial(lagrange2N_1D_factory, N=1)
#: bilinear interpolation is a special case of 2D Lagrange interpolation with N=1
bilinear_interpolation_factory = functools.partial(lagrange2N_2D_factory, N=1)
#: trilinear interpolation is a special case of 3D Lagrange interpolation with N=1
trilinear_interpolation_factory = functools.partial(lagrange2N_3D_factory, N=1)
#: cubic interpolation is a special case of 1D Lagrange interpolation with N=2
cubic_interpolation_factory = functools.partial(lagrange2N_1D_factory, N=2)
#: bicubic interpolation is a special case of 2D Lagrange interpolation with N=2
bicubic_interpolation_factory = functools.partial(lagrange2N_2D_factory, N=2)
#: tricubic interpolation is a special case of 3D Lagrange interpolation with N=2
tricubic_interpolation_factory = functools.partial(lagrange2N_3D_factory, N=2)

#: linear interpolation function for a single particle
linear_interpolation_particle = lagrange2N_1D_particle_factory(N=1)
#: bilinear interpolation function for a single particle
bilinear_interpolation_particle = lagrange2N_2D_particle_factory(N=1)
#: trilinear interpolation function for a single particle
trilinear_interpolation_particle = lagrange2N_3D_particle_factory(N=1)
#: cubic interpolation function for a single particle
cubic_interpolation_particle = lagrange2N_1D_particle_factory(N=2)
#: bicubic interpolation function for a single particle
bicubic_interpolation_particle = lagrange2N_2D_particle_factory(N=2)
#: tricubic interpolation function for a single particle
tricubic_interpolation_particle = lagrange2N_3D_particle_factory(N=2)
