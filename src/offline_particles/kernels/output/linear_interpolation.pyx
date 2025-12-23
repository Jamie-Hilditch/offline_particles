"""Compute outputs by linear interpolation of field  data."""

from functools import partial
from typing import Literal

from .._kernels import ParticleKernel

from cython.parallel cimport prange

from .._core cimport unpack_fielddata_1d, unpack_fielddata_2d, unpack_fielddata_3d
from .._interpolation.linear cimport trilinear_interpolation, bilinear_interpolation, linear_interpolation 


# linear interpolation kernels 

cdef void _linear_interpolation(particles, scalars, fielddata, dimension_idx, field_name, particle_name):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, output
    status = particles.status
    idx = particles[dimension_idx]
    output = particles[particle_name]

    # unpack required field data
    cdef double[::1] field_array
    cdef double off
    field_array, off = unpack_fielddata_1d(fielddata[field_name])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(0, nparticles, schedule='static', nogil=True):

        # skip inactive particles
        if status[i] != 0:
            continue

        # perform linear interpolation
        output[i] = linear_interpolation(
                field_array,
                idx[i] + off
            )

cdef void _bilinear_interpolation(particles, scalars, fielddata, dimension_idx0, dimension_idx1, field_name, particle_name):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] idx0, idx1, output
    status = particles.status
    idx0 = particles[dimension_idx0]
    idx1 = particles[dimension_idx1]
    output = particles[particle_name]

    # unpack required field data
    cdef double[:, ::1] field_array
    cdef double off0, off1
    field_array, off0, off1 = unpack_fielddata_2d(fielddata[field_name])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]
    for i in prange(0, nparticles, schedule='static', nogil=True):
        # skip inactive particles
        if status[i] != 0:
            continue

        # perform bilinear interpolation
        output[i] = bilinear_interpolation(
                field_array,
                idx0[i] + off0,
                idx1[i] + off1
            )

cdef void _trilinear_interpolation(particles, scalars, fielddata field_name, particle_name):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] idx0, idx1, idx2, output
    status = particles.status
    zidx = particles.zidx
    yidx = particles.yidx
    xidx = particles.xidx
    output = particles[particle_name]

    # unpack required field data
    cdef double[:, :, ::1] field_array
    cdef double offz, offy, offx
    field_array, offz, offy, offx = unpack_fielddata_3d(fielddata[field_name])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]
    for i in prange(0, nparticles, schedule='static', nogil=True):
        # skip inactive particles
        if status[i] != 0:
            continue

        # perform trilinear interpolation
        output[i] = trilinear_interpolation(
                field_array,
                zidx[i] + offz,
                yidx[i] + offy,
                xidx[i] + offx
            )

# kernel factories
def linear_interpolation_kernel(field: str, output_name: str | None = None, dimension: Literal["z", "y", "x"] = "z") -> ParticleKernel:
    """Return a ParticleKernel that performs linear interpolation of field data."""
    if dimension not in {"z", "y", "x"}:
        raise ValueError(f"Invalid dimension: {dimension}. Valid options are 'z', 'y', or 'x'.")
    dimension_idx = dimensions + "idx"
    if output_name is None:
        output_name = field

    kernel_function = partial(
        _linear_interpolation,
        dimension_idx=dimension_idx,
        field_name=field,
        particle_name=output_name
    )
    return ParticleKernel(
        kernel_function,
        particle_fields={
            "status": np.uint8,
            dimension_idx: np.float64,
            output_name: np.float64
        },
        scalars=dict(),
        simulation_fields=[field]
    )

def bilinear_interpolation_kernel(field: str, output_name: str | None = None, dimensions: Literal[("z", "y"), ("z", "x"), ("y", "x")] = ("y", "x")) -> ParticleKernel:
    """Return a ParticleKernel that performs bilinear interpolation of field data."""
    valid_dimensions = {("z", "y"), ("z", "x"), ("y", "x")}
    if dimensions not in valid_dimensions:
        raise ValueError(f"Invalid dimensions: {dimensions}. Valid options are {valid_dimensions}.")
    dimension_idx0 = dimensions[0] + "idx"
    dimension_idx1 = dimensions[1] + "idx"
    if output_name is None:
        output_name = field

    kernel_function = partial(
        _bilinear_interpolation,
        dimension_idx0=dimension_idx0,
        dimension_idx1=dimension_idx1,
        field_name=field,
        particle_name=output_name
    )
    return ParticleKernel(
        kernel_function,
        particle_fields={
            "status": np.uint8,
            dimension_idx0: np.float64,
            dimension_idx1: np.float64,
            output_name: np.float64
        },
        scalars=dict(),
        simulation_fields=[field]
    )

def trilinear_interpolation_kernel(field: str, output_name: str | None = None) -> ParticleKernel:
    """Return a ParticleKernel that performs trilinear interpolation of field data."""
    if output_name is None:
        output_name = field
    kernel_function = partial(
        _trilinear_interpolation,
        field_name=field,
        particle_name=output_name
    )
    return ParticleKernel(
        kernel_function,
        particle_fields={
            "status": np.uint8,
            "zidx": np.float64,
            "yidx": np.float64,
            "xidx": np.float64,
            output_name: np.float64
        },
        scalars=dict(),
        simulation_fields=[field]
    )

def vertical_interpolation_kernel(field: str, output_name: str | None = None) -> ParticleKernel:
    """Return a ParticleKernel that performs vertical linear interpolation of field data."""
    return linear_interpolation_kernel(field, output_name, dimension="z")

def horizontal_interpolation_kernel(field: str, output_name: str | None = None) -> ParticleKernel:
    """Return a ParticleKernel that performs horizontal bilinear interpolation of field data."""
    return bilinear_interpolation_kernel(field, output_name, dimensions=("y", "x"))