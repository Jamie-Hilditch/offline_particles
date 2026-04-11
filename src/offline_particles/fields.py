"""Submodule for handling fields in offline particle simulations."""

import abc
import dataclasses
import logging
import warnings
from typing import TYPE_CHECKING, Any, Mapping

import dask.array as da
import numba
import numpy as np
import numpy.typing as npt

from .spatial_arrays import BBox, ChunkedDaskArray, Dimension, NumpyArray, SpatialArray, Stagger

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class FieldData:
    array: npt.NDArray
    offsets: tuple[float, ...]

    def __repr__(self) -> str:
        return f"FieldData(array(shape={self.array.shape}, dtype={self.array.dtype}), offsets={self.offsets})"


class Field(abc.ABC):
    """Abstract base class for fields used in particle simulations."""

    def __init__(
        self,
        z_stagger: Stagger,
        y_stagger: Stagger,
        x_stagger: Stagger,
        *,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self._z_stagger = z_stagger
        self._y_stagger = y_stagger
        self._x_stagger = x_stagger
        self._z_offset = z_stagger.offset
        self._y_offset = y_stagger.offset
        self._x_offset = x_stagger.offset
        if attrs is None:
            attrs = {}
        self._attrs = attrs

    @property
    def z_stagger(self) -> Stagger:
        """Staggering in the vertical direction."""
        return self._z_stagger

    @property
    def y_stagger(self) -> Stagger:
        """Stagger in the eta (y) direction."""
        return self._y_stagger

    @property
    def x_stagger(self) -> Stagger:
        """Stagger in the xi (x) direction."""
        return self._x_stagger

    @property
    def stagger(self) -> tuple[Stagger, Stagger, Stagger]:
        """Staggering of the (z, y, x) dimensions."""
        return (self._z_stagger, self._y_stagger, self._x_stagger)

    @property
    def attrs(self) -> dict[str, Any]:
        """Attributes associated with the field."""
        return self._attrs

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
    def all_offsets(self) -> tuple[float | None, float | None, float | None]:
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
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """Data type of the field."""
        pass

    @property
    def output_dtype(self) -> np.dtype:
        """Output data type of the field."""
        # Default to the same as the underlying data.
        return self.dtype

    @property
    @abc.abstractmethod
    def spatial_shape(self) -> tuple[int, ...]:
        """Shape of the spatial dimensions of the field."""
        pass

    @property
    @abc.abstractmethod
    def nspatial_dims(self) -> int:
        """Number of spatial dimensions of the field."""
        pass

    @abc.abstractmethod
    def validate_shape(self, simulation_shape: tuple[int, int, int, int]) -> None:
        """Validate that the field's shape is compatible with the domain shape."""
        pass

    @abc.abstractmethod
    def get_field_data(self, time_index: float, bbox: BBox) -> FieldData:
        """Get the field data at a given time index.

         Parameters
        ----------
        time_index : float
            Time index.
        bbox : BBox
            Bounding box to extract data from defined in terms of centered grid indices.

        Returns
        -------
        FieldData
            Namedtuple containing the field data array and offsets.
        """
        pass


class StaticField(Field):
    """Class representing static fields that do not change over time."""

    def __init__(
        self,
        data: SpatialArray,
        *,
        attrs: dict[str, Any] | None = None,
    ):
        super().__init__(
            z_stagger=data.z_stagger,
            y_stagger=data.y_stagger,
            x_stagger=data.x_stagger,
            attrs=attrs,
        )
        self._data = data

    @property
    def data(self) -> SpatialArray:
        """The underlying spatial array data."""
        return self._data

    def __repr__(self) -> str:
        return (
            f"StaticField(shape={self._data.shape}, "
            f"z_stagger={self.z_stagger}, "
            f"y_stagger={self.y_stagger}, "
            f"x_stagger={self.x_stagger})"
        )

    def __str__(self) -> str:
        return f"StaticField on z={self.z_stagger.name}, y={self.y_stagger.name}, x={self.x_stagger.name} grid"

    @property
    def dtype(self) -> np.dtype:
        """Data type of the field."""
        return self._data.dtype

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Shape of the spatial dimensions of the field."""
        return self._data.shape

    @property
    def nspatial_dims(self) -> int:
        """Number of spatial dimensions of the field."""
        return len(self.spatial_shape)

    def validate_shape(self, simulation_shape: tuple[int, int, int, int]) -> None:
        """Validate that the field's shape is compatible with the domain shape."""
        staggered_shape = (
            self.z_stagger.expected_size(simulation_shape[1]),
            self.y_stagger.expected_size(simulation_shape[2]),
            self.x_stagger.expected_size(simulation_shape[3]),
        )
        expected_shape = tuple(s for s in staggered_shape if s is not None)
        if self._data.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape} but data has shape {self._data.shape}")

    def get_field_data(self, time_index: float, bbox: BBox) -> FieldData:
        """Get the field data at a given time index.

        Since this is a static field, the time_index is ignored.

        Parameters
        ----------
        time_index : float
            Time index (ignored for static fields).
        bbox : BBox
            Bounding box to extract data from defined in terms of centered grid indices.

        Returns
        -------
        FieldData
            Namedtuple containing the field data array and offsets.
        """
        # For static fields, we ignore time_index
        array, offsets = self._data.get_data_subset(bbox)
        return FieldData(array, offsets)

    @classmethod
    def from_numpy(
        cls,
        data: npt.NDArray,
        z_stagger: Stagger | str,
        y_stagger: Stagger | str,
        x_stagger: Stagger | str,
        *,
        attrs: dict[str, Any] | None = None,
    ) -> "StaticField":
        """Create a StaticField from a NumPy array."""
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected a NumPy array, got {type(data).__name__}")
        spatial_array = NumpyArray(
            data=data,
            z_stagger=Stagger(z_stagger),
            y_stagger=Stagger(y_stagger),
            x_stagger=Stagger(x_stagger),
        )
        return cls(data=spatial_array, attrs=attrs)

    @classmethod
    def from_dask(
        cls,
        data: da.Array,
        z_stagger: Stagger | str,
        y_stagger: Stagger | str,
        x_stagger: Stagger | str,
        *,
        attrs: dict[str, Any] | None = None,
    ) -> "StaticField":
        """Create a StaticField from a chunked Dask array."""
        if not isinstance(data, da.Array):
            raise TypeError(f"Expected a Dask array, got {type(data).__name__}")
        spatial_array = ChunkedDaskArray(
            data=data,
            z_stagger=Stagger(z_stagger),
            y_stagger=Stagger(y_stagger),
            x_stagger=Stagger(x_stagger),
        )
        return cls(data=spatial_array, attrs=attrs)

    @classmethod
    def from_arraylike(
        cls,
        data: npt.ArrayLike,
        z_stagger: Stagger | str,
        y_stagger: Stagger | str,
        x_stagger: Stagger | str,
        *,
        attrs: dict[str, Any] | None = None,
    ) -> "StaticField":
        """Create a StaticField by converting the input to a NumPy array.

        Notes
        -----
        This method eagerly materializes the input into a NumPy array using
        :func:`numpy.asarray`. For Dask arrays, prefer :meth:`StaticField.from_dask`
        to avoid triggering an unexpected compute.
        """
        if isinstance(data, da.Array):
            warnings.warn(
                "StaticField.from_arraylike received a dask.array.Array and will eagerly compute it. "
                "Use StaticField.from_dask for Dask arrays.",
                stacklevel=2,
            )
        return cls.from_numpy(np.asarray(data), z_stagger, y_stagger, x_stagger, attrs=attrs)


type SpatialArrayFactory = type[NumpyArray] | type[ChunkedDaskArray]


class TimeDependentField(Field):
    """Class representing a time-dependent field."""

    def __init__(
        self,
        data: da.Array | npt.NDArray,
        z_stagger: Stagger | str,
        y_stagger: Stagger | str,
        x_stagger: Stagger | str,
        spatial_array_factory: SpatialArrayFactory = NumpyArray,
        output_dtype: npt.DTypeLike = np.float64,
        *,
        attrs: dict[str, Any] | None = None,
    ):
        super().__init__(
            z_stagger=Stagger(z_stagger),
            y_stagger=Stagger(y_stagger),
            x_stagger=Stagger(x_stagger),
            attrs=attrs,
        )

        if data.ndim < 2:
            raise ValueError(
                "TimeDependentField requires at least 2 dimensions (time + spatial). For spatially invariant fields use a scalar."
            )
        self._data = data
        self._spatial_array_factory = spatial_array_factory
        self._data_dtype = data.dtype
        self._output_dtype = np.dtype(output_dtype)

        # temporary arrays for interpolation
        self._allocate_interpolation_arrays((0,) * (data.ndim - 1))

        # delta cache
        self._cached_delta_valid = False
        self._cached_offsets = (np.nan,) * self.nspatial_dims

        # output cache
        self._output_valid = False
        self._prior_ft_output: np.float64 = np.nan  # ft when output was computed

        # time index
        if self._data.shape[0] < 2:
            raise ValueError("TimeDependentField requires at least 2 time steps.")
        self._num_timesteps = self._data.shape[0]
        self._It = 0
        self._previous_time_slice = self._spatial_array_factory(
            self._data[0, ...], self.z_stagger, self.y_stagger, self.x_stagger
        )
        self._next_time_slice = self._spatial_array_factory(
            self._data[1, ...], self.z_stagger, self.y_stagger, self.x_stagger
        )

    def _allocate_interpolation_arrays(self, shape: tuple[int, ...]) -> None:
        """Allocate temporary arrays for interpolation."""
        self._array_shape = shape
        self._delta = np.empty(shape=shape, dtype=self._data.dtype)
        self._cached_delta_valid = False
        self._output = np.empty(shape=shape, dtype=self._output_dtype)
        self._output_valid = False

    @property
    def data(self) -> da.Array | npt.NDArray:
        """The underlying time-dependent data array."""
        return self._data

    def __repr__(self) -> str:
        return (
            f"TimeDependentField(shape={self._data.shape}, "
            f"z_stagger={self.z_stagger}, "
            f"y_stagger={self.y_stagger}, "
            f"x_stagger={self.x_stagger}, "
            f"spatial_array_factory={self._spatial_array_factory.__name__})"
        )

    def __str__(self) -> str:
        return f"TimeDependentField on z={self.z_stagger.name}, y={self.y_stagger.name}, x={self.x_stagger.name} grid"

    @property
    def dtype(self) -> np.dtype:
        """Data type of the field."""
        return self._data_dtype

    @property
    def output_dtype(self) -> np.dtype:
        """Output data type of the field."""
        return self._output_dtype

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Shape of the spatial dimensions of the field."""
        return self._data.shape[1:]

    @property
    def nspatial_dims(self) -> int:
        """Number of spatial dimensions of the field."""
        return len(self.spatial_shape)

    def validate_shape(self, simulation_shape: tuple[int, int, int, int]) -> None:
        """Validate that the field's shape is compatible with the domain shape."""
        staggered_shape = (
            simulation_shape[0],
            self.z_stagger.expected_size(simulation_shape[1]),
            self.y_stagger.expected_size(simulation_shape[2]),
            self.x_stagger.expected_size(simulation_shape[3]),
        )
        expected_shape = tuple(s for s in staggered_shape if s is not None)
        if self._data.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape} but data has shape {self._data.shape}")

    @property
    def previous_time_slice(self) -> SpatialArray:
        """Get the previous time slice as a SpatialArray."""
        return self._previous_time_slice

    @property
    def next_time_slice(self) -> SpatialArray:
        """Get the next time slice as a SpatialArray."""
        return self._next_time_slice

    def increment_time(self) -> None:
        """Increment the time index, creating the next spatial arrays."""
        # error if at largest time
        if self._It == self._num_timesteps - 2:
            raise IndexError("Cannot increment past the penultimate timestep.")
        self._It += 1
        self._previous_time_slice = self._next_time_slice
        self._next_time_slice = self._spatial_array_factory(
            self._data[self._It + 1, ...],
            self.z_stagger,
            self.y_stagger,
            self.x_stagger,
        )
        # loading new data invalidates cached delta and output
        self._cached_delta_valid = False
        self._output_valid = False

    def decrement_time(self) -> None:
        """Decrement the time index, creating the previous spatial arrays."""
        # error if at smallest time
        if self._It == 0:
            raise IndexError("Cannot decrement past the first timestep.")
        self._It -= 1
        self._next_time_slice = self._previous_time_slice
        self._previous_time_slice = self._spatial_array_factory(
            self._data[self._It, ...],
            self.z_stagger,
            self.y_stagger,
            self.x_stagger,
        )
        # loading new data invalidates cached delta and output
        self._cached_delta_valid = False
        self._output_valid = False

    def set_time_index(self, It: int) -> None:
        """Set the time index, adjusting the spatial arrays."""
        # if previous time index do nothing
        if It == self._It:
            return
        # if it's the next timestep we can increment
        if It == self._It + 1:
            return self.increment_time()
        # if it's the previous timestep we can decrement
        if It == self._It - 1:
            return self.decrement_time()
        # else check range
        if It < 0 or It > self._num_timesteps - 2:
            raise IndexError(f"Valid range of time indices is 0,...,{self._num_timesteps - 2}, got {It}.")

        self._It = It
        self._previous_time_slice = self._spatial_array_factory(
            self._data[self._It, ...], self.z_stagger, self.y_stagger, self.x_stagger
        )
        self._next_time_slice = self._spatial_array_factory(
            self._data[self._It + 1, ...],
            self.z_stagger,
            self.y_stagger,
            self.x_stagger,
        )
        # loading new data invalidates cached delta and output
        self._cached_delta_valid = False
        self._output_valid = False

    def get_field_data(self, time_index: float, bbox: BBox) -> FieldData:
        """Get the field data at a given time index.

         Parameters
        ----------
        time_index : float
            Time index.
        bbox : BBox
            Bounding box to extract data from defined in terms of centered grid indices.

        Returns
        -------
        FieldData
            Namedtuple containing the field data array and offsets.
        """
        It, ft = divmod(time_index, 1)
        It = int(It)

        # first make sure we're at the right time index
        self.set_time_index(It)

        # if ft has changed output is invalid
        if ft != self._prior_ft_output:
            self._output_valid = False

        # get the previous time subset
        # note this is cached if the required chunks have not changed
        previous_data, offsets = self._previous_time_slice.get_data_subset(bbox)

        # check offsets match
        if offsets != self._cached_offsets:
            self._cached_offsets = offsets
            self._cached_delta_valid = False
            self._output_valid = False

        # check array shapes match
        if self._array_shape != previous_data.shape:
            self._allocate_interpolation_arrays(previous_data.shape)

        # load delta
        if not self._cached_delta_valid:
            logger.debug(
                "Calculating delta at time index %d with offsets %s",
                It,
                repr(offsets),
            )
            next_data, _ = self._next_time_slice.get_data_subset(bbox)
            np.subtract(next_data, previous_data, out=self._delta)
            self._cached_delta_valid = True

        # perform interpolation if needed
        if not self._output_valid:
            # perform interpolation in time - note we ravel arrays for numba
            # all these arrays are contiguous so ravel generates 1D views
            _ft = self._data_dtype.type(ft)
            _perform_interpolation(previous_data.ravel(), self._delta.ravel(), _ft, self._output.ravel())

            # store when this output was computed
            self._prior_ft_output = ft
            self._output_valid = True

        return FieldData(self._output, offsets)

    @classmethod
    def from_numpy(
        cls,
        data: npt.NDArray,
        z_stagger: Stagger | str,
        y_stagger: Stagger | str,
        x_stagger: Stagger | str,
        *,
        attrs: dict[str, Any] | None = None,
    ) -> "TimeDependentField":
        """Create a TimeDependentField from a NumPy array."""
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected a NumPy array, got {type(data).__name__}")
        return cls(data, z_stagger, y_stagger, x_stagger, NumpyArray, attrs=attrs)

    @classmethod
    def from_dask(
        cls,
        data: da.Array,
        z_stagger: Stagger | str,
        y_stagger: Stagger | str,
        x_stagger: Stagger | str,
        *,
        preload_space: bool = False,
        attrs: dict[str, Any] | None = None,
    ) -> "TimeDependentField":
        """Create a TimeDependentField from a chunked Dask array."""
        if not isinstance(data, da.Array):
            raise TypeError(f"Expected a Dask array, got {type(data).__name__}")
        if preload_space:
            factory = NumpyArray
        else:
            factory = ChunkedDaskArray
        return cls(data, z_stagger, y_stagger, x_stagger, factory, attrs=attrs)

    @classmethod
    def from_arraylike(
        cls,
        data: npt.ArrayLike,
        z_stagger: Stagger | str,
        y_stagger: Stagger | str,
        x_stagger: Stagger | str,
        *,
        attrs: dict[str, Any] | None = None,
    ) -> "TimeDependentField":
        """Create a TimeDependentField by converting the input to a NumPy array.

        Notes
        -----
        This method eagerly materializes the input into a NumPy array using
        :func:`numpy.asarray`. For Dask arrays, prefer :meth:`TimeDependentField.from_dask`
        to avoid triggering an unexpected compute.
        """
        if isinstance(data, da.Array):
            warnings.warn(
                "TimeDependentField.from_arraylike received a dask.array.Array and will eagerly compute it. "
                "Use TimeDependentField.from_dask for Dask arrays.",
                stacklevel=2,
            )
        return cls.from_numpy(np.asarray(data), z_stagger, y_stagger, x_stagger, attrs=attrs)


type Tin = np.floating
type Tout = np.floating


@numba.njit(parallel=True, fastmath=True, nogil=True)
def _perform_interpolation(
    previous: npt.NDArray[Tin], delta: npt.NDArray[Tin], ft: Tin, output: npt.NDArray[Tout]
) -> None:
    """Perform linear interpolation in time."""
    n = previous.size
    for i in numba.prange(n):  # ty: ignore[not-iterable]
        output[i] = previous[i] + ft * delta[i]


def field_from_dataarray(
    data_array: "xr.DataArray",
    dim_map: Mapping[str, Dimension],
    *,
    preload_space: bool = False,
    attrs: dict[str, Any] | None = None,
) -> Field:
    """Create a :class:`Field` from an :class:`xarray.DataArray`.

    Parameters
    ----------
    data_array:
        The DataArray to convert.
    dim_map:
        Mapping from DataArray dimension names to :class:`~offline_particles.spatial_arrays.Dimension`
        enum values.  Every dimension that appears in *data_array* must have an
        entry.  Entries for dimensions that are not present in *data_array* are
        silently ignored; the corresponding spatial direction is treated as
        :attr:`~offline_particles.spatial_arrays.Stagger.INVARIANT`.
    preload_space:
        For Dask-backed arrays only.  When ``True`` each spatial time-slice is
        loaded into a :class:`~offline_particles.spatial_arrays.NumpyArray`
        instead of a :class:`~offline_particles.spatial_arrays.ChunkedDaskArray`,
        which preloads the data eagerly.  Default ``False``.
    attrs:
        Optional attribute dictionary to attach to the resulting
        :class:`Field`.  When ``None`` the attributes are taken from
        ``data_array.attrs``.

    Returns
    -------
    Field
        A :class:`StaticField` when *data_array* has no time dimension, or a
        :class:`TimeDependentField` when a ``TIME`` dimension is present in
        *dim_map* and in *data_array*.

    Raises
    ------
    TypeError
        If *data_array* is not an :class:`xarray.DataArray`.
    ValueError
        If *data_array* has dimensions that are absent from *dim_map*, or if
        two dimensions in *data_array* map to the same spatial direction.
    """
    import xarray as xr

    if not isinstance(data_array, xr.DataArray):
        raise TypeError(f"Expected an xarray DataArray, got {type(data_array).__name__!r}.")

    da_dims: tuple[str, ...] = tuple(data_array.dims)

    # Every DataArray dimension must be covered by dim_map.
    unmapped = set(da_dims) - set(dim_map)
    if unmapped:
        raise ValueError(
            f"DataArray has dimension(s) {sorted(unmapped)!r} that are not present in dim_map. "
            "Add an entry for each dimension to dim_map."
        )

    # Inherit attrs from the DataArray when none are provided explicitly.
    if attrs is None:
        attrs = dict(data_array.attrs)

    # Determine which dimension corresponds to TIME (at most one allowed).
    time_dim: str | None = None
    for dim_name in da_dims:
        if dim_map[dim_name].is_time:
            if time_dim is not None:
                raise ValueError(
                    f"Multiple dimensions ({time_dim!r} and {dim_name!r}) are both mapped to TIME."
                )
            time_dim = dim_name

    # Map each spatial direction to the corresponding DataArray dim and stagger.
    direction_to_dim: dict[str, str] = {}
    direction_to_stagger: dict[str, Stagger] = {}
    for dim_name in da_dims:
        dim_spec = dim_map[dim_name]
        if dim_spec.is_time:
            continue
        direction: str = dim_spec.direction
        if direction in direction_to_dim:
            raise ValueError(
                f"Dimensions {direction_to_dim[direction]!r} and {dim_name!r} both map to "
                f"direction '{direction}'.  Each direction may only appear once."
            )
        direction_to_dim[direction] = dim_name
        stagger = dim_spec.stagger
        assert stagger is not None  # only TIME has stagger=None; we filtered TIME above
        direction_to_stagger[direction] = stagger

    # Directions absent from this DataArray default to INVARIANT.
    z_stagger: Stagger = direction_to_stagger.get("Z", Stagger.INVARIANT)
    y_stagger: Stagger = direction_to_stagger.get("Y", Stagger.INVARIANT)
    x_stagger: Stagger = direction_to_stagger.get("X", Stagger.INVARIANT)

    # Reorder dimensions to the canonical order: (T?, Z?, Y?, X?)
    ordered_dims: list[str] = []
    if time_dim is not None:
        ordered_dims.append(time_dim)
    for direction in ("Z", "Y", "X"):
        if direction in direction_to_dim:
            ordered_dims.append(direction_to_dim[direction])

    data_array_transposed = data_array.transpose(*ordered_dims)

    # Extract the underlying array, preserving Dask arrays where possible.
    underlying: npt.NDArray | da.Array
    if isinstance(data_array_transposed.data, da.Array):
        underlying = data_array_transposed.data
    else:
        underlying = np.asarray(data_array_transposed)

    # Build the appropriate Field subclass.
    if time_dim is not None:
        if isinstance(underlying, da.Array):
            return TimeDependentField.from_dask(
                underlying,
                z_stagger,
                y_stagger,
                x_stagger,
                preload_space=preload_space,
                attrs=attrs,
            )
        else:
            return TimeDependentField.from_numpy(underlying, z_stagger, y_stagger, x_stagger, attrs=attrs)
    else:
        if isinstance(underlying, da.Array):
            return StaticField.from_dask(underlying, z_stagger, y_stagger, x_stagger, attrs=attrs)
        else:
            return StaticField.from_numpy(underlying, z_stagger, y_stagger, x_stagger, attrs=attrs)
