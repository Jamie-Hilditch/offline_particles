"""Core functions for working with field data."""

import numpy as np

cdef inline tuple unpack_field_data_1d(fd):
    cdef double[::1] array
    cdef double offset0 = fd.offsets[0]
    array = np.ascontiguousarray(fd.array)
    return array, offset0

cdef inline tuple unpack_field_data_2d(fd):
    cdef double[:, ::1] array
    cdef double offset0 = fd.offsets[0]
    cdef double offset1 = fd.offsets[1]
    array = np.ascontiguousarray(fd.array)
    return array, offset0, offset1

cdef inline tuple unpack_field_data_3d(fd):
    cdef double[:, :, ::1] array
    cdef double offset0 = fd.offsets[0]
    cdef double offset1 = fd.offsets[1]
    cdef double offset2 = fd.offsets[2]
    array = np.ascontiguousarray(fd.array)
    return array, offset0, offset1, offset2
