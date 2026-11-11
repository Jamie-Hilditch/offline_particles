"""Interpolation kernels."""

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
from ._lagrange_particle_implementations import (
    lagrange_1D_particle_factory,
    lagrange_2D_particle_factory,
    lagrange_3D_particle_factory,
    lagrange_mapped_particle_factory,
)

__all__ = [
    "bicubic_interpolation_particle",
    "bilinear_interpolation_particle",
    "construct_1D_interpolation_kernel",
    "construct_2D_interpolation_kernel",
    "construct_3D_interpolation_kernel",
    "construct_XYZ_interpolation_kernel",
    "construct_XY_interpolation_kernel",
    "construct_XZY_interpolation_kernel",
    "construct_XZ_interpolation_kernel",
    "construct_X_interpolation_kernel",
    "construct_YXZ_interpolation_kernel",
    "construct_YX_interpolation_kernel",
    "construct_YZX_interpolation_kernel",
    "construct_YZ_interpolation_kernel",
    "construct_Y_interpolation_kernel",
    "construct_ZXY_interpolation_kernel",
    "construct_ZX_interpolation_kernel",
    "construct_ZYX_interpolation_kernel",
    "construct_ZY_interpolation_kernel",
    "construct_Z_interpolation_kernel",
    "cubic_interpolation_particle",
    "lagrange_1D_particle_factory",
    "lagrange_2D_particle_factory",
    "lagrange_3D_particle_factory",
    "lagrange_mapped_particle_factory",
    "linear_interpolation_particle",
    "tricubic_interpolation_particle",
    "trilinear_interpolation_particle",
]


#: linear interpolation function for a single particle
linear_interpolation_particle = lagrange_1D_particle_factory(1)
#: bilinear interpolation function for a single particle
bilinear_interpolation_particle = lagrange_2D_particle_factory(1, 1)
#: trilinear interpolation function for a single particle
trilinear_interpolation_particle = lagrange_3D_particle_factory(1, 1, 1)
#: cubic interpolation function for a single particle
cubic_interpolation_particle = lagrange_1D_particle_factory(2)
#: bicubic interpolation function for a single particle
bicubic_interpolation_particle = lagrange_2D_particle_factory(2, 2)
#: tricubic interpolation function for a single particle
tricubic_interpolation_particle = lagrange_3D_particle_factory(2, 2, 2)
