"""Submodule handling loading and data access for arrays of spatial data."""

import abc
import enum
import itertools

import cachetools
import dask.array as da
import numba
import numpy as np
import numpy.typing as npt

from typing import Literal, Self

# constants
DEFAULT_CACHE_SIZE: int = 10_000
CACHE_TYPES = {
    "STD": cachetools.Cache,
    "FIFO": cachetools.FIFOCache,
    "LRU": cachetools.LRUCache,
    "LFU": cachetools.LFUCache
}

# classes 

@enum.unique
class Stagger(enum.StrEnum):
    """Enumeration of possible grid staggerings for a dimension."""

    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    INNER = "inner"
    OUTER = "outer"
    INVARIANT = "invariant"

    @property
    def offset(self) -> float | None:
        """Offset between centered indices and staggered indices."""
        match self:
            case Stagger.CENTER:
                return 0.0
            case Stagger.LEFT | Stagger.OUTER:
                return 0.5
            case Stagger.RIGHT | Stagger.INNER:
                return -0.5
            case Stagger.INVARIANT:
                return None

    def expected_size(self, N: int) -> int | None:
        """Expected size of dimension given size of centered dimension."""
        match self:
            case Stagger.CENTER | Stagger.LEFT | Stagger.RIGHT:
                return N
            case Stagger.OUTER:
                return N + 1
            case Stagger.INNER:
                return N - 1
            case Stagger.INVARIANT:
                return None

    @property
    def is_invariant(self) -> bool:
        return self is Stagger.INVARIANT

    @property
    def on_face(self) -> bool:
        return self in {Stagger.LEFT, Stagger.RIGHT, Stagger.INNER, Stagger.OUTER}

    @property
    def at_center(self) -> bool:
        return self is Stagger.CENTER


class SpatialArray(abc.ABC):
    """Abstract base class for arrays of spatial data."""

    def __init__(
        self,
        z_stagger: Stagger,
        y_stagger: Stagger,
        x_stagger: Stagger,
    ) -> Self:
        self._z_stagger = z_stagger
        self._y_stagger = y_stagger
        self._x_stagger = x_stagger
        self._z_offset = z_stagger.offset
        self._y_offset = y_stagger.offset
        self._x_offset = x_stagger.offset

    @property
    def z_stagger(self) -> Stagger:
        """Staggering of the z dimension."""
        return self._z_stagger

    @property
    def y_stagger(self) -> Stagger:
        """Staggering of the y dimension."""
        return self._y_stagger

    @property
    def x_stagger(self) -> Stagger:
        """Staggering of the x dimension."""
        return self._x_stagger

    @property
    def stagger(self) -> tuple[Stagger, Stagger, Stagger]:
        """Staggering of the (z, y, x) dimensions."""
        return (self._z_stagger, self._y_stagger, self._x_stagger)

    @property
    def z_offset(self) -> float | None:
        """Offset of the z dimension."""
        return self._z_offset

    @property
    def y_offset(self) -> float | None:
        """Offset of the y dimension."""
        return self._y_offset

    @property
    def x_offset(self) -> float | None:
        """Offset of the x dimension."""
        return self._x_offset

    @property
    def offset(self) -> tuple[float | None, float | None, float | None]:
        """Offset of the (z, y, x) dimensions."""
        return (self._z_offset, self._y_offset, self._x_offset)

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        pass

    def offset_and_split_indices(
        self, particle_indices: npt.NDArray[float]
    ) -> tuple[npt.NDArray[int], npt.NDArray[float]]:
        """Compute offset indices and split into integer and fractional parts.

        Parameters
        ----------
        global_indices : npt.NDArray[float]
            (N,3) Array of global indices where each row corresponds to a point in space
            with respect to the centered grid. The columns correspond to (z, y, x) indices.

        Returns
        -------
        tuple[npt.NDArray[int], npt.NDArray[float]]
            Tuple of (N,M) arrays containing the local integer and fractional indices.
            The local indices account for the staggering of the grid and the possibility that
            the view may be only a subset of the full domain.
        """
        return offset_and_split_indices(
            particle_indices,
            self._z_offset,
            self._y_offset,
            self._x_offset,
            self.shape,
        )

    @abc.abstractmethod
    def get_data_subset(
        self, global_indices: npt.NDArray[int]
    ) -> tuple[npt.NDArray[float], npt.NDArray[int]]:
        """Get a subset of the data around the global indices.

        Parameters
        ----------
        global_indices : npt.NDArray[int]
            (N,M) Array of global indices where N is the number of particles and M <= 3 is the dimensionality
            of the spatial array.

        Returns
        -------
        npt.NDArray[float]
            (N,M) Array of values covering the global indices.
        npt.NDArray[int]
            (N,M) array containing the local integer indices.
            The local indices account for the possibility that the view may be only a subset of the full domain.
        """
        pass

class NumpyArray(SpatialArray):
    """Spatial array backed by a NumPy array."""

    def __init__(
        self,
        data: npt.ArrayLike,
        z_stagger: Stagger,
        y_stagger: Stagger,
        x_stagger: Stagger,
    ) -> Self:
        super().__init__(z_stagger, y_stagger, x_stagger)
        self._data = np.array(data)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        return self._data.shape

    def get_data_subset(
        self, global_indices: npt.NDArray[int]
    ) -> tuple[npt.NDArray[float], npt.NDArray[int]]:
        """Get a view of the data around the global indices.

        Parameters
        ----------
        global_indices : npt.NDArray[int]
            (N,M) Array of global indices where N is the number of particles and M <= 3 is the dimensionality
            of the spatial array.

        Returns
        -------
        npt.NDArray[float]
            (N,M) Array of values covering the global indices.
        npt.NDArray[int]
            (N,M) array containing the local integer indices.
            The local indices account for the possibility that the view may be only a subset of the full domain.
        """
        # Here all the data is in memory so we can just return the full data array and the indices unchanged
        return self._data, global_indices

class CachedDaskArray(SpatialArray):
    """Spatial array backed by a dask array with chunk caching."""

    def __init__(
        self,
        data: da.Array,
        z_stagger: Stagger,
        y_stagger: Stagger,
        x_stagger: Stagger,
        *, 
        cache_type: Literal["STD", "FIFO", "LRU", "LFU"] = "FIFO", 
        cache_size: int | None = None
    ) -> Self:
        super().__init__(z_stagger, y_stagger, x_stagger)
        self._data = data
        self._ndim = self._data.ndim
        self._shape = self._data.shape
        self._chunks = data.chunks
        self._bounds = tuple(np.cumulative_sum(chunk, include_initial=True) for chunk in self._chunks)
        self._chunk_index_fn = CHUNK_INDEX_FUNCTIONS[self._ndim]
        # create cache
        if cache_size is None:
            cache_size = DEFAULT_CACHE_SIZE
        cache_type = CACHE_TYPES.get(cache_type)
        if cache_type is None:
            raise ValueError(f"Unknown cache type. Valid types are {CACHE_TYPES.keys()}")
        self._cache = cache_type(cache_size)


    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        return self._shape

    def _get_chunk(
        self, chunk_idx: tuple[int, ...]
    ) -> npt.NDArray[float]:
        """Get a chunk of data given its chunk index.

        Parameters
        ----------
        chunk_idx : tuple[int, ...]
            M-tuple of integers specifying the chunk index in each dimension.

        Returns
        -------
        npt.NDArray[float]
            Array of data for the specified chunk.
        """
        """Get chunk from cache."""
        try:
            data_chunk = self._cache[chunk_idx]
        except KeyError:
            data_chunk = self._data.blocks[chunk_idx].compute()
            self._cache[chunk_idx] = data_chunk
        return data_chunk

    def get_data_subset(
        self, global_indices: npt.NDArray[int]
    ) -> tuple[npt.NDArray[float], npt.NDArray[int]]:
        """Get a view of the data around the global indices.

        Parameters
        ----------
        global_indices : npt.NDArray[int]
            (N,M) Array of global indices where N is the number of particles and M <= 3 is the dimensionality
            of the spatial array.

        Returns
        -------
        npt.NDArray[float]
            (N,M) Array of values covering the global indices.
        npt.NDArray[int]
            (N,M) array containing the local integer indices.
            The local indices account for the possibility that the view may be only a subset of the full domain.
        """
        # find the extent of the indices we need to load
        lower = np.min(global_indices, axis=0)
        upper = np.max(global_indices, axis=0) + 1  # add 1 since we will need this point to construct vertices

        # find the chunk indices and lower bounds
        lower_chunk_idx, lower_bounds = self._chunk_index_fn(lower, self._bounds)
        upper_chunk_idx, _ = self._chunk_index_fn(upper, self._bounds)

        # allocate the superchunk
        superchunk_shape = tuple(
            self._bounds[dim][upper_chunk_idx[dim] + 1] - self._bounds[dim][lower_chunk_idx[dim]]
            for dim in range(self._ndim)
        )
        superchunk = np.empty(superchunk_shape, dtype=self._data.dtype)

        # iterate over all required chunks and insert into superchunk
        chunk_ranges = [range(lower_chunk_idx[dim], upper_chunk_idx[dim] + 1) for dim in range(self._ndim)]
        for chunk_idx in itertools.product(*chunk_ranges):
            # get the chunk data
            data_chunk = self._get_chunk(chunk_idx)

            # compute the slice into the superchunk
            superchunk_slices = tuple(
                slice(
                    self._bounds[dim][chunk_idx[dim]] - lower_bounds[dim],
                    self._bounds[dim][chunk_idx[dim] + 1] - lower_bounds[dim],
                )
                for dim in range(self._ndim)
            )

            # insert the chunk data into the superchunk
            superchunk[superchunk_slices] = data_chunk

        # compute local indices
        local_indices = global_indices - np.array(lower_bounds, dtype=global_indices.dtype).reshape((1, -1))

        return superchunk, local_indices
        


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

# numba functions for chunk indexing
@numba.jit(nogil=True)
def chunk_index_1d(global_idx: npt.NDArray[int], bounds: tuple[npt.NDArray[int]]) -> tuple[tuple[int], tuple[int]]:
    """Get chunk index and bound for 1D array."""
    c0 = np.searchsorted(bounds[0], global_idx[0], side="right") - 1
    b0 = bounds[0][c0]
    return (c0,), (b0,)

@numba.jit(nogil=True)
def chunk_index_2d(global_idx: npt.NDArray[int], bounds: tuple[npt.NDArray[int], npt.NDArray[int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    c0 = np.searchsorted(bounds[0], global_idx[0], side="right") - 1
    b0 = bounds[0][c0]
    c1 = np.searchsorted(bounds[1], global_idx[1], side="right") - 1
    b1 = bounds[1][c1]
    return (c0, c1), (b0, b1)

@numba.jit(nogil=True)
def chunk_index_3d(global_idx: npt.NDArray[int], bounds: tuple[npt.NDArray[int], npt.NDArray[int], npt.NDArray[int]]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    c0 = np.searchsorted(bounds[0], global_idx[0], side="right") - 1
    b0 = bounds[0][c0]
    c1 = np.searchsorted(bounds[1], global_idx[1], side="right") - 1
    b1 = bounds[1][c1]
    c2 = np.searchsorted(bounds[2], global_idx[2], side="right") - 1
    b2 = bounds[2][c2]
    return (c0, c1, c2), (b0, b1, b2)

# map ndim to functions
CHUNK_INDEX_FUNCTIONS = {
    1: chunk_index_1d,
    2: chunk_index_2d,
    3: chunk_index_3d
}