"""Submodule handling loading and data access for arrays of spatial data."""

import abc
import enum

import dask.array as da
import numba
import numpy as np
import numpy.typing as npt

from typing import Self


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
    def offsets(self) -> tuple[float | None, float | None, float | None]:
        """Offset of the (z, y, x) dimensions."""
        return (self._z_offset, self._y_offset, self._x_offset)

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        pass

    @abc.abstractmethod
    def get_data_subset(
        self, particle_indices: tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]
    ) -> tuple[npt.NDArray[float], tuple[float | None, float | None, float | None]]:
        """Get a view of the data around the particle indices.

        Parameters
        ----------
        particle_indices : tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]
            3-tuple (z,y,x) of (N,) arrays of particle indices where N is the number of particles.
            Particle indices are floats defined relative to the centered grid.

        Returns
        -------
        npt.NDArray[float]
            (N,M) Array of values covering the particles.
        tuple[float | None, float | None, float | None]
            Offsets applied to the particle indices (z, y, x) in order to index into the returned data.
            This accounts for both the grid staggering and any subsetting of the data array.
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
        self, particle_indices: tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]
    ) -> tuple[npt.NDArray[float], tuple[float | None, float | None, float | None]]:
        """Get a view of the data around the particle indices.

        Parameters
        ----------
        particle_indices : tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]
            3-tuple (z,y,x) of (N,) arrays of particle indices where N is the number of particles.
            Particle indices are floats defined relative to the centered grid.

        Returns
        -------
        npt.NDArray[float]
            (N,M) Array of values covering the particles.
        tuple[float | None, float | None, float | None]
            Offsets applied to the particle indices (z, y, x) in order to index into the returned data.
            This accounts for both the grid staggering and any subsetting of the data array.
        """
        # Here all the data is in memory so we can just return the full data array and the indices unchanged
        return self._data, self.offsets


class ChunkedDaskArray(SpatialArray):
    """Spatial array backed by a chunked dask array."""

    def __init__(
        self,
        data: da.Array,
        z_stagger: Stagger,
        y_stagger: Stagger,
        x_stagger: Stagger,
    ) -> Self:
        super().__init__(z_stagger, y_stagger, x_stagger)
        self._data = data
        self._ndim = self._data.ndim
        self._shape = self._data.shape
        self._chunks = data.chunks
        self._bounds = tuple(
            np.cumulative_sum(chunk, include_initial=True) for chunk in self._chunks
        )
        self._subset: npt.NDArray[float] | None = None  # type: ignore[call-arg]
        # initialise lower and upper bounds to larger and smaller than possible values
        self._subset_lower_bounds = np.array(self._shape, dtype=int)
        self._subset_upper_bounds = np.zeros((self._ndim,), dtype=int)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        return self._shape

    def get_data_subset(
        self, particle_indices: tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]
    ) -> tuple[npt.NDArray[float], tuple[float | None, float | None, float | None]]:
        """Get a view of the data around the particle indices.

        Parameters
        ----------
        particle_indices : tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]
            3-tuple (z,y,x) of (N,) arrays of particle indices where N is the number of particles.
            Particle indices are floats defined relative to the centered grid.

        Returns
        -------
        npt.NDArray[float]
            (N,M) Array of values covering the particles.
        tuple[float | None, float | None, float | None]
            Offsets applied to the particle indices (z, y, x) in order to index into the returned data.
            This accounts for both the grid staggering and any subsetting of the data array.
        """
        recompute, offsets = compute_new_bounds(
            particle_indices,
            self.offsets,
            self._subset_lower_bounds,
            self._subset_upper_bounds,
            self._bounds,
        )

        if recompute:
            subset_slices = tuple(
                slice(self._subset_lower_bounds[dim], self._subset_upper_bounds[dim])
                for dim in range(self._ndim)
            )
            self._subset = self._data[subset_slices].compute()

        return self._subset, offsets


@numba.jit(nogil=True, fastmath=True)
def compute_new_bounds(
    particle_indices: tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]],
    offsets: tuple[float | None, float | None, float | None],
    lower: npt.NDArray[int],
    upper: npt.NDArray[int],
    bounds: tuple[npt.NDArray[int], ...],
) -> bool:
    """
    Compute new lower and upper bounds for chunked data access.
    Parameters:
        particle_indices: 3-tuple (z,y,x) of (N,) array of particle indices.
        offsets: tuple of offsets to apply to the indices where M is the number of non-None offsets.
        lower: (M,) array of lower bounds.
        upper: (M,) array of upper bounds.
        bounds: tuple of M arrays containing the chunk boundaries for each dimension.
    Returns:
        - bool indicating whether the bounds have changed.
        - offsets applied to the particle indices.
    """
    bounds_changed = False
    dim = 0

    out_offsets = []

    for i in range(3):
        if offsets[i] is None:
            out_offsets.append(None)
            continue
        # apply offset to particle indices
        global_lower = np.min(particle_indices[dim]) + offsets[i]
        global_upper = np.max(particle_indices[dim]) + offsets[i] + 1  # add 1 for upper bound

        # find chunk-aligned bounds
        lower_bound = _lower_chunk_bound(global_lower, bounds[dim])
        upper_bound = _upper_chunk_bound(global_upper, bounds[dim])

        if lower_bound != lower[dim]:
            lower[dim] = lower_bound
            bounds_changed = True
        if upper_bound != upper[dim]:
            upper[dim] = upper_bound
            bounds_changed = True

        out_offsets.append(offsets[i] - lower_bound)

        dim += 1

    return bounds_changed, out_offsets


# numba functions for chunk indexing
@numba.jit(nogil=True)
def _lower_chunk_bound(global_idx: float, bounds: npt.NDArray[int]) -> int:
    """Get the bound of a chunk satisfying b <= global_idx."""
    idx = np.searchsorted(bounds, global_idx, side="right") - 1
    return bounds[idx]


@numba.jit(nogil=True)
def _upper_chunk_bound(global_idx: float, bounds: npt.NDArray[int]) -> int:
    """Get the bound of a chunk satisfying b >= global_idx."""
    idx = np.searchsorted(bounds, global_idx, side="left")
    return bounds[idx]
