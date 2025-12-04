"""Submodule for advecting particles."""

import numba
import numpy as np
import numpy.typing as npt


@numba.jit(parallel=True, nogil=True, fastmath=True)
def _offset_indices(
    idx: npt.NDArray[float],
    offset_z: float | None,
    offset_y: float | None,
    offset_x: float | None,
    out: npt.NDArray[float],
) -> None:
    """Compute offset indices.
    Parameters:
        idx: (N,3) array of floats where N is the number of particles
        offset_z: index offset in the z direction
        offset_y: index offset in the y direction
        offset_x: index offset in the x direction
        out: (N,M) array of floats where M is the number of non-None offsets.
    """
    col = 0
    if offset_z is not None:
        out[:, col] = idx[:, 0] + offset_z
        col += 1
    if offset_y is not None:
        out[:, col] = idx[:, 1] + offset_y
        col += 1
    if offset_x is not None:
        out[:, col] = idx[:, 2] + offset_x
        col += 1


@numba.jit(parallel=True, nogil=True, fastmath=True)
def _split_indices(
    idx: npt.NDArray[float], shape: tuple[int, ...]
) -> tuple[npt.NDArray[int], npt.NDArray[float]]:
    """Split indices into integer and fractional parts.
    Parameters:
        idx: (N,M) array of floats where N is the number of particles and M is the number of dimensions.
        shape: M-tuple containing the shape of underlying data array
    Returns:
        tuple of:
            - (N,M) array of integer indices
            - (N,M) array of fractional indices
    """
    N, M = idx.shape
    int_idx = np.empty((N, M), dtype=np.int32)
    frac_idx = np.empty_like(idx)

    for dim in range(M):
        # find integer part, limited to penultimate index
        np.floor(idx[:, dim], out=int_idx[:, dim], casting="unsafe")
        np.clip(int_idx[:, dim], 0, shape[dim] - 2, out=int_idx[:, dim])
        # compute fractional part as remainder
        frac_idx[:, dim] = idx[:, dim] - int_idx[:, dim]

    return int_idx, frac_idx


@numba.jit(parallel=True, nogil=True, fastmath=True)
def offset_and_split_indices(
    idx: npt.NDArray[float],
    offset_z: float | None,
    offset_y: float | None,
    offset_x: float | None,
    shape: tuple[int, ...],
) -> tuple[npt.NDArray[int], npt.NDArray[float]]:
    """Compute offset indices and split into integer and fractional parts.
    Parameters:
        idx: (N,3) array of floats where N is the number of particles
        offset_z: index offset in the z direction
        offset_y: index offset in the y direction
        offset_x: index offset in the x direction
        shape: M-tuple containing the shape of underlying data array
    Returns:
        tuple of:
            - (N,M) array of integer indices
            - (N,M) array of fractional indices
    """
    N = idx.shape[0]
    M = len(shape)

    offset_idx = np.empty((N, M), dtype=idx.dtype)
    _offset_indices(idx, offset_z, offset_y, offset_x, offset_idx)
    return _split_indices(offset_idx, shape)
