"""Submodule handling loading and data access for arrays of spatial data."""

import abc
import dataclasses
import enum
import logging
from typing import Iterable

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

    @property
    def offset(self) -> float:
        """Offset between centered indices and staggered indices."""
        match self:
            case Stagger.CENTER:
                return 0.0
            case Stagger.LEFT | Stagger.OUTER:
                return 0.5
            case Stagger.RIGHT | Stagger.INNER:
                return -0.5

    def expected_size(self, N: int) -> int:
        """Get the expected size of dimension given size of centered dimension.

        Parameters
        ----------
        N (int)
            Size of the centered dimension.

        Returns
        -------
        int
            Expected size of the dimension with this staggering.
        """
        match self:
            case Stagger.CENTER | Stagger.LEFT | Stagger.RIGHT:
                return N
            case Stagger.OUTER:
                return N + 1
            case Stagger.INNER:
                return N - 1

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
        axis (ArrayAxis | str)
            Either an existing ``ArrayAxis`` member, a canonical value (``"Z"``, ``"Y"``, ``"X"``),
            or an alias name (e.g. ``"DEPTH"``, ``"LATITUDE"``, ``"LON"``).

        Returns
        -------
        ArrayAxis
            The corresponding ``ArrayAxis`` member.

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

    @property
    def particle_index_name(self) -> str:
        """Return the name of the particle property containing the indices for this axis."""
        match self:
            case ArrayAxis.Z:
                return "zidx"
            case ArrayAxis.Y:
                return "yidx"
            case ArrayAxis.X:
                return "xidx"


class ArrayLayout:
    """Specification of a spatial array's axes and staggering."""

    __slots__ = ("ndim", "axes", "staggers", "offsets")

    # Type annotations for the type checker
    ndim: int
    axes: tuple[ArrayAxis, ...]
    staggers: tuple[Stagger, ...]
    offsets: tuple[float, ...]

    def __init__(self, axes: Iterable[ArrayAxis | str], staggers: Iterable[Stagger | str]) -> None:
        axes = tuple(ArrayAxis.parse(axis) for axis in axes)
        staggers = tuple(Stagger(s) for s in staggers)

        # validation
        if len(axes) != len(staggers):
            raise ValueError("Number of axes and staggers must match")
        if len(set(axes)) != len(axes):
            raise ValueError("Axes must be unique")

        # set attributes
        object.__setattr__(self, "ndim", len(axes))
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "staggers", staggers)
        object.__setattr__(self, "offsets", tuple(s.offset for s in staggers))

    def __setattr__(self, name, value):
        raise AttributeError("ArrayLayout is immutable")

    def __delattr__(self, name):
        raise AttributeError("ArrayLayout is immutable")


@dataclasses.dataclass(frozen=True, slots=True)
class BBox:
    """Bounding box defined by min and max indices in each dimension."""

    zmin: float
    zmax: float
    ymin: float
    ymax: float
    xmin: float
    xmax: float

    def axis_bounds(self, axis: ArrayAxis) -> tuple[float, float]:
        """Get the bounding box limits for a specific axis.

        Parameters
        ----------
        axis (ArrayAxis)
            The axis for which to retrieve the bounding box limits.

        Returns
        -------
        tuple[float, float]
            A tuple containing the minimum and maximum bounds for the specified axis.

        Raises
        ------
        ValueError
            If the provided axis is not one of the recognized ArrayAxis values.
        """
        match axis:
            case ArrayAxis.Z:
                return self.zmin, self.zmax
            case ArrayAxis.Y:
                return self.ymin, self.ymax
            case ArrayAxis.X:
                return self.xmin, self.xmax
            case _:
                raise ValueError(f"Invalid axis: {axis}")


class SpatialArray(abc.ABC):
    """Abstract base class for arrays of spatial data."""

    def __init__(
        self,
        layout: ArrayLayout,
    ) -> None:
        self._layout = layout

    @property
    def layout(self) -> ArrayLayout:
        """Layout specification of the spatial array."""
        return self._layout

    @property
    def ndim(self) -> int:
        """Number of dimensions in the spatial array."""
        return self._layout.ndim

    @property
    def axes(self) -> tuple[ArrayAxis, ...]:
        """Axes of the spatial array."""
        return self._layout.axes

    @property
    def staggers(self) -> tuple[Stagger, ...]:
        """Staggering of the dimensions."""
        return self._layout.staggers

    @property
    def offsets(self) -> tuple[float, ...]:
        """Offsets for all dimensions."""
        return self._layout.offsets

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
        layout: ArrayLayout,
    ) -> None:
        super().__init__(layout)
        self._data = np.asarray(data)

        if self._data.ndim != self.layout.ndim:
            raise ValueError(
                f"Data array has {self._data.ndim} dimensions but layout specifies {self.layout.ndim} dimensions."
            )

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
        return self._data, self.offsets


class ChunkedDaskArray(SpatialArray):
    """Spatial array backed by a chunked dask array."""

    def __init__(
        self,
        data: da.Array,
        layout: ArrayLayout,
    ) -> None:
        super().__init__(layout)
        if not isinstance(data, da.Array):
            raise TypeError(f"Expected a Dask array, got {type(data).__name__}")
        if data.ndim != self.layout.ndim:
            raise ValueError(f"Data array has {data.ndim} dimensions but {self.layout.ndim} dimensions were specified.")
        self._data = data
        self._shape = self._data.shape
        self._chunks = data.chunks
        self._chunk_boundaries = tuple(np.cumulative_sum(chunk, include_initial=True) for chunk in self._chunks)
        # placeholders for array and bounds of current subset
        self._subset: npt.NDArray[np.generic] = np.zeros((0,) * data.ndim, data.dtype)
        self._subset_bounds: tuple[tuple[int, int], ...] = ((0, 0),) * self.ndim

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
        offsets = self.offsets

        # get bounding box by axis
        axis_bounds = (bounding_box.axis_bounds(axis) for axis in self.axes)

        # compute new bounds for each axis based on the bounding box, stagger offsets, and chunk boundaries
        new_bounds = tuple(
            _compute_new_bounds(axis_bound, offset, bounds)
            for axis_bound, offset, bounds in zip(axis_bounds, offsets, self._chunk_boundaries)
        )

        # compute new offsets given thee new bounds
        new_offsets = tuple(offset - lb for offset, (lb, _) in zip(offsets, new_bounds))

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
    """Compute new dimension bounds for chunked data access.

    Parameters
    ----------
    dim_bounds (tuple[float, float])
        The minimum and maximum bounds of the dimension.
    offset (float)
        The offset to apply to the indices based on the staggering of the grid.
    bounds (npt.NDArray[np.int_])
        The array containing the chunk boundaries for the dimension.

    Returns
    -------
    tuple[int, int]
        The new lower and upper bounds for the dimension, clamped to the chunk boundaries.
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

    Parameters
    ----------
    dim_min (float)
        The minimum bound of the dimension.
    offset (float)
        The offset to apply to the indices based on the staggering of the grid.
    bounds (npt.NDArray[np.int_])
        The array containing the chunk boundaries for the dimension.

    Returns
    -------
    int
        The new lower bound for the dimension, clamped to the chunk boundaries.
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

    Parameters
    ----------
    dim_max (float)
        The maximum bound of the dimension.
    offset (float)
        The offset to apply to the indices based on the staggering of the grid.
    bounds (npt.NDArray[np.int_])
        The array containing the chunk boundaries for the dimension.

    Returns
    -------
    int
        The new upper bound for the dimension, clamped to the chunk boundaries.
    """
    global_upper = dim_max + offset + 1  # add 1 for upper bound

    # clamp to chunk boundaries
    idx = np.searchsorted(bounds, global_upper, side="left")
    idx = min(len(bounds) - 1, idx)
    return bounds[idx]
