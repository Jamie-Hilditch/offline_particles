"""Kernel functions for linear interpolation of field data."""

from cython.parallel cimport prange

from .._core.inputs cimport unpack_field_data_1d, unpack_field_data_2d, unpack_field_data_3d
from .._core.interpolation.linear cimport (
    trilinear_interpolation,
    bilinear_interpolation,
    linear_interpolation
)
from ..status cimport STATUS

cdef void _linear_interpolation_kernel_function(particle_properties, field_data):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] idx, output
    status = particle_properties["status"]
    idx = particle_properties["idx"]
    output = particle_properties["output"]

    # unpack required field data
    cdef double[::1] field_array
    cdef double off
    field_array, off = unpack_field_data_1d(field_data["field"])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(0, nparticles, schedule='static', nogil=True):

        # skip inactive particles
        if status[i] & STATUS.INACTIVE:
            continue

        # perform linear interpolation
        output[i] = linear_interpolation(
                field_array,
                idx[i] + off
            )

cdef void _bilinear_interpolation_kernel_function(particle_properties, field_data):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] idx_0, idx_1, output
    status = particle_properties["status"]
    idx_0 = particle_properties["idx_0"]
    idx_1 = particle_properties["idx_1"]
    output = particle_properties["output"]

    # unpack required field data
    cdef double[:, ::1] field_array
    cdef double off_0, off_1
    field_array, off_0, off_1 = unpack_field_data_2d(field_data["field"])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(0, nparticles, schedule='static', nogil=True):

        # skip inactive particles
        if status[i] & STATUS.INACTIVE:
            continue

        # perform bilinear interpolation
        output[i] = bilinear_interpolation(
                field_array,
                idx_0[i] + off_0,
                idx_1[i] + off_1
            )

cdef void _trilinear_interpolation_kernel_function(particle_properties, field_data):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx, output
    status = particle_properties["status"]
    zidx = particle_properties["zidx"]
    yidx = particle_properties["yidx"]
    xidx = particle_properties["xidx"]
    output = particle_properties["output"]

    # unpack required field data
    cdef double[:, :, ::1] field_array
    cdef double off_z, off_y, off_x
    field_array, off_z, off_y, off_x = unpack_field_data_3d(field_data["field"])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(0, nparticles, schedule='static', nogil=True):

        # skip inactive particles
        if status[i] & STATUS.INACTIVE:
            continue

        # perform trilinear interpolation
        output[i] = trilinear_interpolation(
                field_array,
                zidx[i] + off_z,
                yidx[i] + off_y,
                xidx[i] + off_x
            )

cdef void _linear_interpolation_accumulation_kernel_function(particle_properties, field_data):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] idx, output
    status = particle_properties["status"]
    idx = particle_properties["idx"]
    output = particle_properties["output"]

    # unpack required field data
    cdef double[::1] field_array
    cdef double off
    field_array, off = unpack_field_data_1d(field_data["field"])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(0, nparticles, schedule='static', nogil=True):

        # skip inactive particles
        if status[i] & STATUS.INACTIVE:
            continue

        # perform linear interpolation
        output[i] += linear_interpolation(
                field_array,
                idx[i] + off
            )

cdef void _bilinear_interpolation_accumulation_kernel_function(particle_properties, field_data):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] idx_0, idx_1, output
    status = particle_properties["status"]
    idx_0 = particle_properties["idx_0"]
    idx_1 = particle_properties["idx_1"]
    output = particle_properties["output"]

    # unpack required field data
    cdef double[:, ::1] field_array
    cdef double off_0, off_1
    field_array, off_0, off_1 = unpack_field_data_2d(field_data["field"])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(0, nparticles, schedule='static', nogil=True):

        # skip inactive particles
        if status[i] & STATUS.INACTIVE:
            continue

        # perform bilinear interpolation
        output[i] += bilinear_interpolation(
                field_array,
                idx_0[i] + off_0,
                idx_1[i] + off_1
            )

cdef void _trilinear_interpolation_accumulation_kernel_function(particle_properties, field_data):
    # unpack required particle fields
    cdef unsigned char[::1] status
    cdef double[::1] zidx, yidx, xidx, output
    status = particle_properties["status"]
    zidx = particle_properties["zidx"]
    yidx = particle_properties["yidx"]
    xidx = particle_properties["xidx"]
    output = particle_properties["output"]

    # unpack required field data
    cdef double[:, :, ::1] field_array
    cdef double off_z, off_y, off_x
    field_array, off_z, off_y, off_x = unpack_field_data_3d(field_data["field"])

    # loop over particles
    cdef Py_ssize_t i, nparticles
    nparticles = status.shape[0]

    for i in prange(0, nparticles, schedule='static', nogil=True):

        # skip inactive particles
        if status[i] & STATUS.INACTIVE:
            continue

        # perform trilinear interpolation
        output[i] += trilinear_interpolation(
                field_array,
                zidx[i] + off_z,
                yidx[i] + off_y,
                xidx[i] + off_x
            )

# python wrappers
cpdef linear_interpolation_kernel_function(particle_properties, scalars, field_data):
    _linear_interpolation_kernel_function(particle_properties, field_data)

cpdef bilinear_interpolation_kernel_function(particle_properties, scalars, field_data):
    _bilinear_interpolation_kernel_function(particle_properties, field_data)

cpdef trilinear_interpolation_kernel_function(particle_properties, scalars, field_data):
    _trilinear_interpolation_kernel_function(particle_properties, field_data)

cpdef linear_interpolation_accumulation_kernel_function(particle_properties, scalars, field_data):
    _linear_interpolation_accumulation_kernel_function(particle_properties, field_data)

cpdef bilinear_interpolation_accumulation_kernel_function(particle_properties, scalars, field_data):
    _bilinear_interpolation_accumulation_kernel_function(particle_properties, field_data)

cpdef trilinear_interpolation_accumulation_kernel_function(particle_properties, scalars, field_data):
    _trilinear_interpolation_accumulation_kernel_function(particle_properties, field_data)
