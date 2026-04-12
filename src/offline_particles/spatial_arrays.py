"""Submodule handling loading and data access for arrays of spatial data."""

import abc
import dataclasses
import enum
import logging

import dask.array as da
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


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
    def is_active(self) -> bool:
        return self is not Stagger.INVARIANT

    @property
    def on_face(self) -> bool:
        return self in {Stagger.LEFT, Stagger.RIGHT, Stagger.INNER, Stagger.OUTER}

    @property
    def at_center(self) -> bool:
        return self is Stagger.CENTER


# convenience definitions of Staggers
ALL_STAGGERS = frozenset(Stagger)
CENTERED_STAGGERS = frozenset({Stagger.CENTER})
ON_FACE_STAGGERS = frozenset({s for s in Stagger if s.on_face})
ACTIVE_STAGGERS = frozenset({s for s in Stagger if s.is_active})
INVARIANT_STAGGERS = frozenset({Stagger.INVARIANT})
INACTIVE_STAGGERS = frozenset({s for s in Stagger if not s.is_active})


class ArrayAxis(enum.StrEnum):
    """Enumeration of possible axes for spatial arrays."""

    # Z axis
    Z = "Z"
    DEPTH = "Z"
    VERTICAL = "Z"

    # Y axis
    Y = "Y"
    LATITUDE = "Y"
    LAT = "Y"
    MERIDIONAL = "Y"

    # X axis
    X = "X"
    LONGITUDE = "X"
    LON = "X"
    ZONAL = "X"

    @classmethod
    def parse(cls, axis: "ArrayAxis | str") -> "ArrayAxis":
        """Return an ``ArrayAxis`` member from a member, canonical value, or alias name.

        Parameters
        ----------
        axis : ArrayAxis | str
            Either an existing ``ArrayAxis`` member, a canonical value (``"Z"``, ``"Y"``, ``"X"``),
            or an alias name (e.g. ``"DEPTH"``, ``"LATITUDE"``, ``"LON"``).

        Raises
        ------
        ValueError
            If the string does not match any ``ArrayAxis`` value or name.
        """
        if isinstance(axis, cls):
            return axis
        # Try canonical value lookup first ("Z", "Y", "X")
        try:
            return cls(axis)
        except ValueError:
            pass
        # Fall back to name lookup to support aliases ("DEPTH", "LATITUDE", etc.)
        try:
            return cls[axis]
        except KeyError:
            pass
        raise ValueError(f"'{axis}' is not a valid ArrayAxis value or name")


@dataclasses.dataclass(frozen=True, slots=True)
class BBox:
    """Bounding box defined by min and max indices in each dimension."""

    zmin: float
    zmax: float
    ymin: float
    ymax: float
    xmin: float
    xmax: float

    @property
    def by_dimension(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Bounding box organized by dimension."""
        return (
            (self.zmin, self.zmax),
            (self.ymin, self.ymax),
            (self.xmin, self.xmax),
        )


class SpatialArray(abc.ABC):
    """Abstract base class for arrays of spatial data."""

    def __init__(
        self,
        z_stagger: Stagger,
        y_stagger: Stagger,
        x_stagger: Stagger,
    ) -> None:
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
    def dmask(self) -> tuple[bool, bool, bool]:
        """Mask indicating which dimensions are active."""
        return (
            self._z_stagger.is_active,
            self._y_stagger.is_active,
            self._x_stagger.is_active,
        )

    @property
    def active_offsets(self) -> tuple[float, ...]:
        """Offsets for active dimensions only."""
        return tuple(
            offset
            for offset, is_active in zip((self._z_offset, self._y_offset, self._x_offset), self.dmask)
            if is_active
        )

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """Data type of the underlying data array."""
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        pass

    @abc.abstractmethod
    def get_data_subset(
        self,
        bounding_box: BBox,
    ) -> tuple[npt.NDArray, tuple[float, ...]]:
        """Get a view of the data around the particle indices.

        Parameters
        ----------
        bounding_box : tuple[float, float, float, float, float, float]
            6-tuple (z_min, z_max, y_min, y_max, x_min, x_max) defining the bounding box of particle indices
            where z,y,x are floats defined relative to the centered grid.

        Returns
        -------
        npt.NDArray[float]
            (N,M) Array of values covering the particles.
        tuple[float, ...]
            Offsets to apply to the active particle indices in order to index into the returned data.
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
    ) -> None:
        super().__init__(z_stagger, y_stagger, x_stagger)
        self._data = np.array(data)

    @property
    def dtype(self) -> np.dtype:
        """Data type of the underlying data array."""
        return self._data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        return self._data.shape

    def get_data_subset(
        self,
        bounding_box: BBox,
    ) -> tuple[npt.NDArray, tuple[float, ...]]:
        """Get a view of the data around the particle indices.

        Parameters
        ----------
        bounding_box : tuple[float, float, float, float, float, float]
            6-tuple (z_min, z_max, y_min, y_max, x_min, x_max) defining the bounding box of particle indices
            where z,y,x are floats defined relative to the centered grid.

        Returns
        -------
        npt.NDArray[float]
            (N,M) Array of values covering the particles.
        tuple[float, ...]
            Offsets to apply to the active particle indices in order to index into the returned data.
            This accounts for both the grid staggering and any subsetting of the data array.
        """
        # Here all the data is in memory so we can just return the full data array and the indices unchanged
        return self._data, self.active_offsets


class ChunkedDaskArray(SpatialArray):
    """Spatial array backed by a chunked dask array."""

    def __init__(
        self,
        data: da.Array,
        z_stagger: Stagger,
        y_stagger: Stagger,
        x_stagger: Stagger,
    ) -> None:
        super().__init__(z_stagger, y_stagger, x_stagger)
        if not isinstance(data, da.Array):
            raise TypeError(f"Expected a Dask array, got {type(data).__name__}")
        if data.ndim != sum(self.dmask):
            raise ValueError(
                f"Data array has {data.ndim} dimensions but {sum(self.dmask)} active dimensions were specified."
            )
        self._data = data
        self._ndim = self._data.ndim
        self._shape = self._data.shape
        self._chunks = data.chunks
        self._bounds = tuple(np.cumulative_sum(chunk, include_initial=True) for chunk in self._chunks)
        # placeholders for array and bounds of current subset
        self._subset: npt.NDArray[np.generic] = np.zeros((0,) * data.ndim, data.dtype)
        self._subset_bounds: tuple[tuple[int, int], ...] = ((0, 0),) * self._ndim

    @property
    def dtype(self) -> np.dtype:
        """Data type of the underlying data array."""
        return self._data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        return self._shape

    def get_data_subset(
        self,
        bounding_box: BBox,
    ) -> tuple[npt.NDArray, tuple[float, ...]]:
        """Get a view of the data around the particle indices.

        Parameters
        ----------
        bounding_box : tuple[float, float, float, float, float, float]
            6-tuple (z_min, z_max, y_min, y_max, x_min, x_max) defining the bounding box of particle indices
            where z,y,x are floats defined relative to the centered grid.

        Returns
        -------
        npt.NDArray[float]
            (N,M) Array of values covering the particles.
        tuple[float, ...]
            Offsets to apply to the active particle indices in order to index into the returned data.
            This accounts for both the grid staggering and any subsetting of the data array.
        """
        # loop through active dimensions and compute new bounds
        active_bbox = tuple(
            dim_bounds for dim_bounds, is_active in zip(bounding_box.by_dimension, self.dmask) if is_active
        )
        active_offsets = self.active_offsets
        new_bounds = tuple(
            _compute_new_bounds(db, offset, bounds)
            for db, offset, bounds in zip(active_bbox, active_offsets, self._bounds)
        )
        new_offsets = tuple(offset - lb for offset, (lb, _) in zip(active_offsets, new_bounds))

        # if new bounds don't match existing update and load new subset
        if self._subset_bounds != new_bounds:
            logger.debug("Loading new data subset with bounds: %s", new_bounds)
            self._subset_bounds = new_bounds
            subset_slices = tuple(slice(*bounds) for bounds in new_bounds)
            self._subset = self._data[subset_slices].compute()  # type: ignore[call-arg]

        return self._subset, new_offsets


def _compute_new_bounds(
    dim_bounds: tuple[float, float], offset: float, bounds: npt.NDArray[np.int_]
) -> tuple[int, int]:
    """
    Compute new dimension bounds for chunked data access.
    """
    dim_min, dim_max = dim_bounds
    new_lower = compute_new_lower_bound(dim_min, offset, bounds)
    new_upper = compute_new_upper_bound(dim_max, offset, bounds)
    return new_lower, new_upper


def compute_new_lower_bound(
    dim_min: float,
    offset: float,
    bounds: npt.NDArray[np.int_],
) -> int:
    """
    Compute new lower bound for chunked data access.
    Parameters:
        dim_min: lower bound of the bounding box.
        offset: offset to apply to the indices.
        bounds: array containing the chunk boundaries.
    Returns:
        - int: lower bound.
    """
    global_lower = dim_min + offset

    # clamp to chunk boundaries
    idx = np.searchsorted(bounds, global_lower, side="right") - 1
    idx = max(0, idx)
    return bounds[idx]


def compute_new_upper_bound(
    dim_max: float,
    offset: float,
    bounds: npt.NDArray[np.int_],
) -> int:
    """
    Compute new upper bound for chunked data access.
    Parameters:
        dim_max: upper bound of the bounding box.
        offset: offset to apply to the indices.
        bounds: array containing the chunk boundaries.
    Returns:
        - int: upper bound.
    """
    global_upper = dim_max + offset + 1  # add 1 for upper bound

    # clamp to chunk boundaries
    idx = np.searchsorted(bounds, global_upper, side="left")
    idx = min(len(bounds) - 1, idx)
    return bounds[idx]
