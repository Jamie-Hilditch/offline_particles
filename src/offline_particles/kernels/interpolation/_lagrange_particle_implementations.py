"""Implementations of Lagrange polynomial interpolation for single particles.

This submodule provides both 1D, 2D, 3D interpolation for single particles and also mapped interpolation.
The mapped functions accept all three particle indices in the order (`zidx`, `yidx`, `xidx`) and then map them
to the correct order for interpolation based on the field array layout, calling the 1D, 2D, or 3D implementation.
This allows us to use the same interpolation functions for different field array layouts
without needing to write separate interpolation functions for each layout.
"""

import functools
from collections.abc import Callable

import numba
import numpy as np
import numpy.typing as npt

from ...spatial_arrays import ArrayAxis
from ._lagrange_basis import lagrange_basis

_AXIS_ENUMERATION = {ArrayAxis.Z: 0, ArrayAxis.Y: 1, ArrayAxis.X: 2}


@numba.njit(nogil=True, fastmath=True)
def _truncate_index(idx: float, max_idx: int) -> int:
    """Truncate the index to be within the bounds of the field array.

    Parameters
    ----------
    idx : float
        The index to truncate.

    max_idx : int
        The maximum index allowed (inclusive).

    Returns
    -------
    int
        The truncated index, which will be in the range [0, `max_idx`].
    """
    idx = int(idx)  # floor the index to get the lower index
    if idx < 0:
        return 0
    elif idx > max_idx:
        return max_idx
    else:
        return idx


@functools.cache
def lagrange_1D_particle_factory(
    N: int, D: int = 0
) -> Callable[[npt.NDArray[np.inexact], np.float64, int], np.inexact]:
    """Create a function for 1D Lagrange polynomial interpolation of a single particle.

    Parameters
    ----------
    N : int
        The number of points on either side of the particle to use for interpolation. The total number of points used for interpolation will be `2N`.
    D : int, optional
        The derivative order. (Default 0)

    Returns
    -------
    Callable
        A JIT compiled function implementing the 1D Lagrange polynomial interpolation for a single particle.

    Raises
    ------
    ValueError
        If `N` is not a positive integer.
        If `D` is not a non-negative integer.

    Notes
    -----
    This factory function is cached such that the same function will be returned for the same inputs. This saves recompiling the function if reused.

    Warning
    -------
    This function is intended to be used in hot loops and thus does not check if `max_idx` is compatible with the field array size.
    I.e. 0 <= `max_idx` <= `field.shape[0] - 1` must hold to avoid out-of-bounds memory access.
    It is the caller's responsibility to ensure that the field array has at least `2N` points to avoid out-of-bounds memory access.
    """
    basis = lagrange_basis(N, D)

    @numba.njit(nogil=True, fastmath=True)
    def lagrange_1D_particle(
        field_array: npt.NDArray[np.inexact],
        offset_idx: np.float64,
        max_idx: int,  # max index for the lower index to avoid out-of-bounds
    ) -> np.inexact:
        """Implement Lagrange polynomial interpolation in 1D for a single particle.

        Parameters
        ----------
        field_array : npt.NDArray[np.inexact]
            The 1D array of field values to interpolate from.
        offset_idx : np.float64
            The offset index for the particle.
        max_idx : int
            The maximum index allowed (inclusive).

        Returns
        -------
        np.inexact
            The interpolated value at the particle's position.
        """
        # work in the field array's dtype to avoid unnecessary casts and preserve precision
        scalar_t = field_array.dtype.type

        # get integer and fractional parts of the index; shift by N-1 (not N) so the 2N-point
        # stencil straddles offset_idx symmetrically: N points at-or-below the lower index, N above
        shifted_idx = offset_idx - (N - 1)
        I0 = _truncate_index(shifted_idx, max_idx)
        x0 = (N - 1) + shifted_idx - I0

        # accumulate value
        value = scalar_t(0.0)
        l0 = basis(x0)
        for i in range(2 * N):
            l0_i = scalar_t(l0[i])
            value += l0_i * field_array[I0 + i]
        return value

    return lagrange_1D_particle


@functools.cache
def lagrange_2D_particle_factory(
    N0: int,
    N1: int,
    D0: int = 0,
    D1: int = 0,
) -> Callable[[npt.NDArray[np.inexact], np.float64, np.float64, int, int], np.inexact]:
    """Create a function for 2D Lagrange polynomial interpolation of a single particle.

    Parameters
    ----------
    N0 : int
        The number of points on either side of the particle to use for interpolation in dimension 0. The total number of points used for interpolation will be 2 * N0.
    N1 : int
        The number of points on either side of the particle to use for interpolation in dimension 1. The total number of points used for interpolation will be 2 * N1.
    D0 : int, optional
        The derivative order in dimension 0. (Default 0)
    D1 : int, optional
        The derivative order in dimension 1. (Default 0)

    Returns
    -------
    Callable
        A JIT compiled function implementing 2D Lagrange polynomial interpolation for a single particle.

    Raises
    ------
    ValueError
        If either `N0` or `N1` is not a positive integer.
        If either `D0` or `D1` is not a non-negative integer.

    Notes
    -----
    This factory function is cached such that the same function will be returned for the same inputs. This saves recompiling the function if reused.

    Warning
    -------
    This function is intended to be used in hot loops and thus does not check if `max_idx_{i}` is compatible with the field array size.
    I.e. 0 <= `max_idx_{i}` <= `field.shape[i] - 1` must hold to avoid out-of-bounds memory access.
    It is the caller's responsibility to ensure that the field array has at least (`2 N0`) x (`2 N1`) points to avoid out-of-bounds memory access.
    """
    basis_0 = lagrange_basis(N0, D0)
    basis_1 = lagrange_basis(N1, D1)

    @numba.njit(nogil=True, fastmath=True)
    def lagrange_2D_particle(
        field_array: npt.NDArray[np.inexact],
        offset_idx_0: np.float64,
        offset_idx_1: np.float64,
        max_idx_0: int,
        max_idx_1: int,
    ) -> np.inexact:
        """Implement a Lagrange interpolating polynomial in 2D for a single particle.

        Parameters
        ----------
        field_array : npt.NDArray[np.inexact]
            The 2D array of field values to interpolate from.
        offset_idx_0 : np.float64
            The offset index for the particle in the first dimension.
        offset_idx_1 : np.float64
            The offset index for the particle in the second dimension.
        max_idx_0 : int
            The maximum index allowed (inclusive) in the first dimension.
        max_idx_1 : int
            The maximum index allowed (inclusive) in the second dimension.

        Returns
        -------
        np.inexact
            The interpolated value at the particle's position.
        """
        # work in the field array's dtype to avoid unnecessary casts and preserve precision
        scalar_t = field_array.dtype.type

        # get integer and fractional parts of the index; shift by N-1 so the 2N-point
        # stencil straddles the target symmetrically: N points at-or-below the lower index, N above
        shifted_idx_0 = offset_idx_0 - (N0 - 1)
        shifted_idx_1 = offset_idx_1 - (N1 - 1)
        I0 = _truncate_index(shifted_idx_0, max_idx_0)
        I1 = _truncate_index(shifted_idx_1, max_idx_1)
        x0 = (N0 - 1) + shifted_idx_0 - I0
        x1 = (N1 - 1) + shifted_idx_1 - I1

        # compute lagrange basis polynomial weights
        l0 = basis_0(x0)
        l1 = basis_1(x1)

        # accumulate value
        value = scalar_t(0.0)
        for i0 in range(2 * N0):
            l0_i = scalar_t(l0[i0])
            for i1 in range(2 * N1):
                l1_j = scalar_t(l1[i1])
                value += l0_i * l1_j * field_array[I0 + i0, I1 + i1]
        return value

    return lagrange_2D_particle


@functools.cache
def lagrange_3D_particle_factory(
    N0: int,
    N1: int,
    N2: int,
    D0: int = 0,
    D1: int = 0,
    D2: int = 0,
) -> Callable[[npt.NDArray[np.inexact], np.float64, np.float64, np.float64, int, int, int], np.inexact]:
    """Create a function for 3D Lagrange polynomial interpolation of a single particle.

    Parameters
    ----------
    N0 : int
        The number of points on either side of the particle to use for interpolation in dimension 0. The total number of points used for interpolation will be 2 * N0.
    N1 : int
        The number of points on either side of the particle to use for interpolation in dimension 1. The total number of points used for interpolation will be 2 * N1.
    N2 : int
        The number of points on either side of the particle to use for interpolation in dimension 2. The total number of points used for interpolation will be 2 * N2.
    D0 : int, optional
        The derivative order in dimension 0. (Default 0)
    D1 : int, optional
        The derivative order in dimension 1. (Default 0)
    D2 : int, optional
        The derivative order in dimension 2. (Default 0)

    Returns
    -------
    Callable
        A JIT compiled function implementing 3D Lagrange polynomial interpolation for a single particle.

    Raises
    ------
    ValueError
        If any of `N0`, `N1`, or `N2` is not a positive integer.
        If any of `D0`, `D1`, or `D2` is not a non-negative integer.

    Notes
    -----
    This factory function is cached such that the same function will be returned for the same inputs. This saves recompiling the function if reused.

    Warning
    -------
    This function is intended to be used in hot loops and thus does not check if `max_idx_{i}` is compatible with the field array size.
    I.e. 0 <= `max_idx_{i}` <= `field.shape[i] - 1` must hold to avoid out-of-bounds memory access.
    It is the caller's responsibility to ensure that the field array has at least (`2 N0`) x (`2 N1`) x (`2 N2`) points to avoid out-of-bounds memory access.
    """
    basis_0 = lagrange_basis(N0, D0)
    basis_1 = lagrange_basis(N1, D1)
    basis_2 = lagrange_basis(N2, D2)

    @numba.njit(nogil=True, fastmath=True)
    def lagrange_3D_particle(
        field_array: npt.NDArray[np.inexact],
        offset_idx_0: np.float64,
        offset_idx_1: np.float64,
        offset_idx_2: np.float64,
        max_idx_0: int,
        max_idx_1: int,
        max_idx_2: int,
    ) -> np.inexact:
        """Implement Lagrange polynomial interpolation in 3D for a single particle.

        Parameters
        ----------
        field_array : npt.NDArray[np.inexact]
            The 3D array of field values to interpolate from.
        offset_idx_0 : np.float64
            The offset index for the particle in the first dimension.
        offset_idx_1 : np.float64
            The offset index for the particle in the second dimension.
        offset_idx_2 : np.float64
            The offset index for the particle in the third dimension.
        max_idx_0 : int
            The maximum index allowed (inclusive) in the first dimension.
        max_idx_1 : int
            The maximum index allowed (inclusive) in the second dimension.
        max_idx_2 : int
            The maximum index allowed (inclusive) in the third dimension.

        Returns
        -------
        np.inexact
            The interpolated value at the particle's position.
        """
        # work in the field array's dtype to avoid unnecessary casts and preserve precision
        scalar_t = field_array.dtype.type

        # get integer and fractional parts of the index; shift by N-1 so the 2N-point
        # stencil straddles the target symmetrically: N points at-or-below the lower index, N above
        shifted_idx_0 = offset_idx_0 - (N0 - 1)
        shifted_idx_1 = offset_idx_1 - (N1 - 1)
        shifted_idx_2 = offset_idx_2 - (N2 - 1)
        I0 = _truncate_index(shifted_idx_0, max_idx_0)
        I1 = _truncate_index(shifted_idx_1, max_idx_1)
        I2 = _truncate_index(shifted_idx_2, max_idx_2)
        x0 = (N0 - 1) + shifted_idx_0 - I0
        x1 = (N1 - 1) + shifted_idx_1 - I1
        x2 = (N2 - 1) + shifted_idx_2 - I2

        # compute lagrange basis polynomial weights
        l0 = basis_0(x0)
        l1 = basis_1(x1)
        l2 = basis_2(x2)

        # accumulate value
        value = scalar_t(0.0)
        for i0 in range(2 * N0):
            l0_i = scalar_t(l0[i0])
            for i1 in range(2 * N1):
                l1_j = scalar_t(l1[i1])
                for i2 in range(2 * N2):
                    l2_k = scalar_t(l2[i2])
                    value += l0_i * l1_j * l2_k * field_array[I0 + i0, I1 + i1, I2 + i2]
        return value

    return lagrange_3D_particle


@functools.cache
def _lagrange_1D_mapped_particle_factory(axis_index: int, N: int, D: int) -> Callable:
    """Create a 1D field interpolator for the given axis and interpolation stencil size.

    Parameters
    ----------
    axis_index : int
        The index of the axis in (`zidx`, `yidx`, `xidx`).
    N : int
        The half-width of the interpolation stencil.
    D : int
        The derivative order.

    Returns
    -------
    Callable
        A 1D field interpolator.

    Raises
    ------
    ValueError
        If `N` is not a positive integer.
        If `D` is not a non-negative integer.
    """
    interpolator = lagrange_1D_particle_factory(N, D)

    @numba.njit(nogil=True, fastmath=True)
    def lagrange_1D_mapped_particle(
        zidx: np.float64,
        yidx: np.float64,
        xidx: np.float64,
        field_array: npt.NDArray[np.inexact],
        offset: float,
        max_idx: int,
    ):
        idx = (zidx, yidx, xidx)[axis_index]
        return interpolator(field_array, idx + offset, max_idx)

    return lagrange_1D_mapped_particle


@functools.cache
def _lagrange_2D_mapped_particle_factory(
    axis_index_0: int, axis_index_1: int, N0: int, N1: int, D0: int, D1: int
) -> Callable:
    """Create a 2D field interpolator for the given axes and interpolation stencil sizes.

    Parameters
    ----------
    axis_index_0 : int
        The index of the first axis in (`zidx`, `yidx`, `xidx`).
    axis_index_1 : int
        The index of the second axis in (`zidx`, `yidx`, `xidx`).
    N0 : int
        The half-width of the interpolation stencil in dimension 0.
    N1 : int
        The half-width of the interpolation stencil in dimension 1.
    D0 : int
        The derivative order in dimension 0.
    D1 : int
        The derivative order in dimension 1.

    Returns
    -------
    Callable
        A 2D field interpolator.

    Raises
    ------
    ValueError
        If either `N0` or `N1` is not a positive integer.
        If either `D0` or `D1` is not a non-negative integer.
    """
    interpolator = lagrange_2D_particle_factory(N0, N1, D0, D1)

    @numba.njit(nogil=True, fastmath=True)
    def lagrange_2D_mapped_particle(
        zidx: np.float64,
        yidx: np.float64,
        xidx: np.float64,
        field_array: npt.NDArray[np.inexact],
        offset_0: float,
        offset_1: float,
        max_idx_0: int,
        max_idx_1: int,
    ):
        idx0 = (zidx, yidx, xidx)[axis_index_0]
        idx1 = (zidx, yidx, xidx)[axis_index_1]
        return interpolator(field_array, idx0 + offset_0, idx1 + offset_1, max_idx_0, max_idx_1)

    return lagrange_2D_mapped_particle


@functools.cache
def _lagrange_3D_mapped_particle_factory(
    axis_index_0: int, axis_index_1: int, axis_index_2: int, N0: int, N1: int, N2: int, D0: int, D1: int, D2: int
) -> Callable:
    """Create a 3D field interpolator for the given axes and interpolation stencil sizes.

    Parameters
    ----------
    axis_index_0 : int
        The index of the first axis in (`zidx`, `yidx`, `xidx`).
    axis_index_1 : int
        The index of the second axis in (`zidx`, `yidx`, `xidx`).
    axis_index_2 : int
        The index of the third axis in (`zidx`, `yidx`, `xidx`).
    N0 : int
        The half-width of the interpolation stencil in dimension 0.
    N1 : int
        The half-width of the interpolation stencil in dimension 1.
    N2 : int
        The half-width of the interpolation stencil in dimension 2.
    D0 : int
        The derivative order in dimension 0.
    D1 : int
        The derivative order in dimension 1.
    D2 : int
        The derivative order in dimension 2.

    Returns
    -------
    Callable
        A 3D field interpolator.

    Raises
    ------
    ValueError
        If any of `N0`, `N1`, or `N2` is not a positive integer.
        If any of `D0`, `D1`, or `D2` is not a non-negative integer.
    """
    interpolator = lagrange_3D_particle_factory(N0, N1, N2, D0, D1, D2)

    @numba.njit(nogil=True, fastmath=True)
    def lagrange_3D_mapped_particle(
        zidx: np.float64,
        yidx: np.float64,
        xidx: np.float64,
        field_array: npt.NDArray[np.inexact],
        offset_0: float,
        offset_1: float,
        offset_2: float,
        max_idx_0: int,
        max_idx_1: int,
        max_idx_2: int,
    ):
        idx0 = (zidx, yidx, xidx)[axis_index_0]
        idx1 = (zidx, yidx, xidx)[axis_index_1]
        idx2 = (zidx, yidx, xidx)[axis_index_2]
        return interpolator(
            field_array, idx0 + offset_0, idx1 + offset_1, idx2 + offset_2, max_idx_0, max_idx_1, max_idx_2
        )

    return lagrange_3D_mapped_particle


@functools.cache
def lagrange_mapped_particle_factory(
    dim_ordering: tuple[ArrayAxis, ...], N: tuple[int, ...], D: tuple[int, ...] | None = None
) -> Callable:
    """Create a field interpolator for the given field array layout and interpolation stencil size.

    Parameters
    ----------
    dim_ordering : tuple[ArrayAxis, ...]
        The ordering of dimensions in the field array.
    N : tuple[int, ...]
        The half-width of the interpolation stencil in each dimension.
    D : tuple[int, ...] | None, optional
        The derivative order in each dimension. If `None` defaults to a tuple of 0s.

    Returns
    -------
    Callable
        A field interpolator.

    Raises
    ------
    ValueError
        If there are duplicate dimensions in `dim_ordering`.
        If the number of dimensions in the field array is not supported.
        If any element of `N` is not a positive integer.
        If any element of `D` is not a non-negative integer.
    """
    # check the dim_ordering is unique
    if len(set(dim_ordering)) != len(dim_ordering):
        raise ValueError(
            f"Duplicate dimensions in dim_ordering: {dim_ordering}. Each dimension should only appear once."
        )

    # resolve None derivative order
    if D is None:
        D = tuple(0 for _ in N)

    # tuple of indices for the axes in the field array, numba should treat this as a compile time constant
    axis_indices = tuple(_AXIS_ENUMERATION[axis] for axis in dim_ordering)

    match axis_indices, N, D:
        case (A,), (N,), (D,):
            return _lagrange_1D_mapped_particle_factory(A, N, D)
        case (A0, A1), (N0, N1), (D0, D1):
            return _lagrange_2D_mapped_particle_factory(A0, A1, N0, N1, D0, D1)
        case (A0, A1, A2), (N0, N1, N2), (D0, D1, D2):
            return _lagrange_3D_mapped_particle_factory(A0, A1, A2, N0, N1, N2, D0, D1, D2)
        case _:
            raise ValueError(
                f"Unsupported or inconsistent number of dimensions: {len(axis_indices) = }, {len(N) = }, {len(D) = }. Supported number of dimensions are 1, 2, or 3."
            )
