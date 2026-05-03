"""Functions for interpolation."""

cdef inline Py_ssize_t truncate_index(double idx, Py_ssize_t max_idx) noexcept nogil:
    cdef Py_ssize_t i = <Py_ssize_t>idx
    if i < 0:
        i = 0
    elif i > max_idx:
        i = max_idx
    return i

cdef inline double linear_interpolation(const double[::1] array, double idx) noexcept nogil:
    cdef Py_ssize_t I0 = truncate_index(idx, array.shape[0] - 2)

    cdef double f0 = idx - I0
    cdef double g0 = 1.0 - f0

    return g0 * array[I0] + f0 * array[I0 + 1]

cdef inline double bilinear_interpolation(const double[:, ::1] array, double idx0, double idx1) noexcept nogil:
    cdef Py_ssize_t I0 = truncate_index(idx0, array.shape[0] - 2)
    cdef Py_ssize_t I1 = truncate_index(idx1, array.shape[1] - 2)

    cdef double f0 = idx0 - I0
    cdef double f1 = idx1 - I1
    cdef double g0 = 1.0 - f0
    cdef double g1 = 1.0 - f1

    cdef double v00 = array[I0, I1]
    cdef double v01 = array[I0, I1 + 1]
    cdef double v10 = array[I0 + 1, I1]
    cdef double v11 = array[I0 + 1, I1 + 1]

    return g0 * g1 * v00 + g0 * f1 * v01 + f0 * g1 * v10 + f0 * f1 * v11

cdef inline double trilinear_interpolation(
    const double[:, :, ::1] array, double idx0, double idx1, double idx2
) noexcept nogil:
    cdef Py_ssize_t I0 = truncate_index(idx0, array.shape[0] - 2)
    cdef Py_ssize_t I1 = truncate_index(idx1, array.shape[1] - 2)
    cdef Py_ssize_t I2 = truncate_index(idx2, array.shape[2] - 2)

    cdef double f0 = idx0 - I0
    cdef double f1 = idx1 - I1
    cdef double f2 = idx2 - I2
    cdef double g0 = 1.0 - f0
    cdef double g1 = 1.0 - f1
    cdef double g2 = 1.0 - f2

    cdef double v000 = array[I0, I1, I2]
    cdef double v001 = array[I0, I1, I2 + 1]
    cdef double v010 = array[I0, I1 + 1, I2]
    cdef double v011 = array[I0, I1 + 1, I2 + 1]
    cdef double v100 = array[I0 + 1, I1, I2]
    cdef double v101 = array[I0 + 1, I1, I2 + 1]
    cdef double v110 = array[I0 + 1, I1 + 1, I2]
    cdef double v111 = array[I0 + 1, I1 + 1, I2 + 1]

    return (g0 * g1 * g2 * v000 +
            g0 * g1 * f2 * v001 +
            g0 * f1 * g2 * v010 +
            g0 * f1 * f2 * v011 +
            f0 * g1 * g2 * v100 +
            f0 * g1 * f2 * v101 +
            f0 * f1 * g2 * v110 +
            f0 * f1 * f2 * v111)
