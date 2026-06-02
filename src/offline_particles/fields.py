"""Submodule for handling fields in offline particle simulations."""

import abc
import dataclasses
import logging
import warnings
from collections.abc import Mapping
from typing import Any

import dask.array as da
import numba
import numpy as np
import numpy.typing as npt
import xarray as xr

from .spatial_arrays import ArrayAxis, ArrayLayout, BBox, ChunkedDaskArray, NumpyArray, SpatialArray, Stagger

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class FieldData:
    array: npt.NDArray
    offsets: tuple[float, ...]

    def __repr__(self) -> str:
        return f"FieldData(array(shape={self.array.shape}, dtype={self.array.dtype}), offsets={self.offsets})"

    def unpack(self) -> tuple[npt.NDArray, *tuple[float, ...]]:
        """Unpack the FieldData into its components.

        Returns
        -------
        tuple[npt.NDArray, float, ...]
            A tuple containing the field data array followed by the offsets for each dimension.
        """
        return self.array, *self.offsets


@dataclasses.dataclass(frozen=True, slots=True)
class SimulationSize:
    time: int
    z: int
    y: int
    x: int

    def axis_size(self, axis: ArrayAxis) -> int:
        match axis:
            case ArrayAxis.Z:
                return self.z
            case ArrayAxis.Y:
                return self.y
            case ArrayAxis.X:
                return self.x
            case _:
                raise ValueError(f"Invalid axis: {axis}")


class Field(abc.ABC):
    """Abstract base class for fields used in particle simulations."""

    def __init__(
        self,
        layout: ArrayLayout,
        *,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self._layout = layout
        if attrs is None:
            attrs = {}
        self._attrs = attrs

    @property
    def layout(self) -> ArrayLayout:
        """The array layout of the field."""
        return self._layout

    @property
    def ndim(self) -> int:
        """Number of dimensions in the spatial array."""
        return self.layout.ndim

    @property
    def axes(self) -> tuple[ArrayAxis, ...]:
        """Axes of the spatial array."""
        return self.layout.axes

    @property
    def staggers(self) -> tuple[Stagger, ...]:
        """Staggering of the dimensions."""
        return self.layout.staggers

    @property
    def offsets(self) -> tuple[float, ...]:
        """Offsets for all dimensions."""
        return self.layout.offsets

    @property
    def attrs(self) -> dict[str, Any]:
        """Attributes associated with the field."""
        return self._attrs

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """Data type of the field."""

    @property
    def output_dtype(self) -> np.dtype:
        """Output data type of the field."""
        # Default to the same as the underlying data.
        return self.dtype

    @property
    @abc.abstractmethod
    def spatial_shape(self) -> tuple[int, ...]:
        """Shape of the spatial dimensions of the field."""

    @property
    def nspatial_dims(self) -> int:
        """Number of spatial dimensions of the field."""
        return self._layout.ndim

    @abc.abstractmethod
    def validate_shape(self, simulation_size: SimulationSize) -> None:
        """Validate that the field's shape is compatible with the sizes of the simulation dimensions."""

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


class StaticField(Field):
    """Class representing static fields that do not change over time."""

    def __init__(
        self,
        data: SpatialArray,
        *,
        attrs: dict[str, Any] | None = None,
    ):
        super().__init__(
            layout=data.layout,
            attrs=attrs,
        )
        if data.layout.ndim < 1:
            raise ValueError(
                "StaticField requires at least 1 spatial dimension. For spatially invariant fields use a scalar."
            )
        self._data = data

    @property
    def data(self) -> SpatialArray:
        """The underlying spatial array data."""
        return self._data

    def __repr__(self) -> str:
        return f"StaticField(shape={self._data.shape}, dtype={self.dtype}, layout={self.layout})"

    def __str__(self) -> str:
        return f"StaticField on {self._data.layout} grid"

    @property
    def dtype(self) -> np.dtype:
        """Data type of the field."""
        return self._data.dtype

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Shape of the spatial dimensions of the field."""
        return self._data.shape

    def validate_shape(self, simulation_size: SimulationSize) -> None:
        """Validate that the field's shape is compatible with the sizes of the simulation dimensions.

        Parameters
        ----------
        simulation_size : SimulationSize
            The sizes of the simulation dimensions to validate against.

        Raises
        ------
        ValueError
            If the field's shape does not match the expected sizes based on the simulation dimensions and staggers.
        """
        for data_size, axis, stagger in zip(self._data.shape, self.axes, self.staggers):
            simulation_axis_size = simulation_size.axis_size(axis)
            expected_size = stagger.expected_size(simulation_axis_size)
            if data_size != expected_size:
                raise ValueError(f"Expected size {expected_size} along axis {axis} but got {data_size}")

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
            Dataclass containing the field data array and offsets.
        """
        # For static fields, we ignore time_index
        array, offsets = self._data.get_data_subset(bbox)
        return FieldData(array, offsets)

    @classmethod
    def from_numpy(
        cls,
        data: npt.NDArray,
        axes: tuple[ArrayAxis | str, ...],
        staggers: tuple[Stagger | str, ...],
        *,
        attrs: dict[str, Any] | None = None,
    ) -> "StaticField":
        """Create a StaticField from a NumPy array.

        Parameters
        ----------
        data : npt.NDArray
            Input NumPy array containing the field data.
        axes : tuple[ArrayAxis | str, ...]
            Tuple of axes corresponding to the spatial dimensions of the field.
        staggers : tuple[Stagger | str, ...]
            Tuple of staggers corresponding to the spatial dimensions of the field.
        attrs : dict[str, Any] | None, optional
            Attributes for the field.

        Returns
        -------
        StaticField
            A StaticField instance created from the input NumPy array.

        Raises
        ------
        TypeError
            If the input data is not a NumPy array.
        """
        layout = ArrayLayout(axes, staggers)
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected a NumPy array, got {type(data).__name__}")
        spatial_array = NumpyArray(
            data=data,
            layout=layout,
        )
        return cls(data=spatial_array, attrs=attrs)

    @classmethod
    def from_dask(
        cls,
        data: da.Array,
        axes: tuple[ArrayAxis | str, ...],
        staggers: tuple[Stagger | str, ...],
        *,
        attrs: dict[str, Any] | None = None,
    ) -> "StaticField":
        """Create a StaticField from a chunked Dask array.

        Parameters
        ----------
        data : da.Array
            Input Dask array containing the field data.
        axes : tuple[ArrayAxis | str, ...]
            Tuple of axes corresponding to the spatial dimensions of the field.
        staggers : tuple[Stagger | str, ...]
            Tuple of staggers corresponding to the spatial dimensions of the field.
        attrs : dict[str, Any] | None, optional
            Attributes for the field.

        Returns
        -------
        StaticField
            A StaticField instance created from the input Dask array.

        Raises
        ------
        TypeError
            If the input data is not a Dask array.
        """
        layout = ArrayLayout(axes, staggers)
        if not isinstance(data, da.Array):
            raise TypeError(f"Expected a Dask array, got {type(data).__name__}")
        spatial_array = ChunkedDaskArray(
            data=data,
            layout=layout,
        )
        return cls(data=spatial_array, attrs=attrs)

    @classmethod
    def from_arraylike(
        cls,
        data: npt.ArrayLike,
        axes: tuple[ArrayAxis | str, ...],
        staggers: tuple[Stagger | str, ...],
        *,
        attrs: dict[str, Any] | None = None,
    ) -> "StaticField":
        """Create a StaticField by converting the input to a NumPy array.

        Parameters
        ----------
        data : npt.ArrayLike
            Input data that can be converted to a NumPy array.
        axes : tuple[ArrayAxis | str, ...]
            Tuple of axes corresponding to the spatial dimensions of the field.
        staggers : tuple[Stagger | str, ...]
            Tuple of staggers corresponding to the spatial dimensions of the field.
        attrs : dict[str, Any] | None, optional
            Attributes for the field.

        Returns
        -------
        StaticField
            A StaticField instance created from the input data.

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
        return cls.from_numpy(np.asarray(data), axes, staggers, attrs=attrs)

    @classmethod
    def from_xarray(
        cls,
        data: xr.DataArray,
        dims: Mapping[str, tuple[ArrayAxis | str, Stagger | str]],
        *,
        ignore_missing_dims: bool = False,
    ) -> "StaticField":
        """Create a StaticField from an xarray DataArray.

        Parameters
        ----------
        data : xr.DataArray
            The input xarray DataArray containing the field data.
        dims : Mapping[str, tuple[ArrayAxis | str, Stagger | str]]
            Mapping of dimension names to ``(ArrayAxis, Stagger)`` tuples.
        ignore_missing_dims : bool, optional
            If True, dimensions specified in ``dims`` that are not present in the DataArray will
            be ignored. If False (default), a ValueError will be raised.

        Returns
        -------
        StaticField
            A StaticField instance created from the input xarray DataArray.

        Raises
        ------
        ValueError
            If any of the dimensions are not found in the dataset dimensions.
            If the size of a centered dimension is not provided and the centered dimension is not included in dims.
            If a dimension has a size that does not match the expected size based on the stagger and provided size.

        Notes
        -----
        The dimension names provided in ``dims`` must match the dimensions of ``data`` exactly unless
        ``ignore_missing_dims`` is True, in which case extra dimensions in ``dims`` that are not present
        in ``data`` will be ignored. All dimensions in ``data`` must be accounted for in ``dims``
        regardless of the value of ``ignore_missing_dims``.
        """
        # build an array layout from the provided dims
        dims_mapping = dict(dims)  # make a copy to avoid mutating the input
        axes = []
        staggers = []
        for dim in data.dims:
            if dim not in dims:
                raise ValueError(f"Dimension '{dim}' in data is missing from dims mapping.")
            axis, stagger = dims_mapping.pop(dim)
            axes.append(ArrayAxis.parse(axis))
            staggers.append(Stagger(stagger))

        # error if there are any extra dimensions in dims that were not in data
        if (not ignore_missing_dims) and dims_mapping:
            raise ValueError(f"Dimensions in dims mapping not found in data: {list(dims_mapping.keys())}")

        array = data.data

        # create spatial array
        if isinstance(array, da.Array):
            return cls.from_dask(
                data=array,
                axes=tuple(axes),
                staggers=tuple(staggers),
                attrs=data.attrs,
            )
        else:
            return cls.from_arraylike(
                data=array,
                axes=tuple(axes),
                staggers=tuple(staggers),
                attrs=data.attrs,
            )


type SpatialArrayFactory = type[NumpyArray | ChunkedDaskArray]


class TimeDependentField(Field):
    """Class representing a time-dependent field."""

    def __init__(
        self,
        data: da.Array | npt.NDArray,
        layout: ArrayLayout,
        spatial_array_factory: SpatialArrayFactory = NumpyArray,
        output_dtype: npt.DTypeLike | None = None,
        *,
        attrs: dict[str, Any] | None = None,
    ):
        super().__init__(
            layout=layout,
            attrs=attrs,
        )

        if data.ndim < 2:
            raise ValueError(
                "TimeDependentField requires at least 2 dimensions (time + spatial). For spatially invariant fields use a scalar."
            )
        self._data = data
        self._spatial_array_factory = spatial_array_factory
        self._data_dtype = data.dtype
        if output_dtype is None:
            output_dtype = self._data_dtype
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
        self._previous_time_slice = self._spatial_array_factory(self._data[0, ...], self.layout)
        self._next_time_slice = self._spatial_array_factory(self._data[1, ...], self.layout)

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
            f"dtype={self._data.dtype}, "
            f"layout={self.layout}, "
            f"spatial_array_factory={self._spatial_array_factory.__name__})"
        )

    def __str__(self) -> str:
        return f"TimeDependentField on {self.layout} grid"

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

    def validate_shape(self, simulation_size: SimulationSize) -> None:
        """Validate that the field's shape is compatible with the sizes of the simulation dimensions.

        Parameters
        ----------
        simulation_size : SimulationSize
            The sizes of the simulation dimensions to validate against.

        Raises
        ------
        ValueError
            If the field's shape does not match the expected sizes based on the simulation dimensions and staggers.
        """
        # first validate time dimension
        if self._data.shape[0] != simulation_size.time:
            raise ValueError(f"Expected size {simulation_size.time} along time axis but got {self._data.shape[0]}")
        for data_size, axis, stagger in zip(self._data.shape[1:], self.axes, self.staggers):
            simulation_axis_size = simulation_size.axis_size(axis)
            expected_size = stagger.expected_size(simulation_axis_size)
            if data_size != expected_size:
                raise ValueError(f"Expected size {expected_size} along axis {axis} but got {data_size}")

    @property
    def previous_time_slice(self) -> SpatialArray:
        """Get the previous time slice as a SpatialArray."""
        return self._previous_time_slice

    @property
    def next_time_slice(self) -> SpatialArray:
        """Get the next time slice as a SpatialArray."""
        return self._next_time_slice

    def increment_time(self) -> None:
        """Increment the time index, creating the next spatial arrays.

        Raises
        ------
        IndexError
            If the time index is already at the penultimate timestep and cannot be incremented further.
        """
        # error if at largest time
        if self._It == self._num_timesteps - 2:
            raise IndexError("Cannot increment past the penultimate timestep.")
        self._It += 1
        self._previous_time_slice = self._next_time_slice
        self._next_time_slice = self._spatial_array_factory(self._data[self._It + 1, ...], self.layout)
        # loading new data invalidates cached delta and output
        self._cached_delta_valid = False
        self._output_valid = False

    def decrement_time(self) -> None:
        """Decrement the time index, creating the previous spatial arrays.

        Raises
        ------
        IndexError
            If the time index is already at the first timestep and cannot be decremented further.
        """
        # error if at smallest time
        if self._It == 0:
            raise IndexError("Cannot decrement past the first timestep.")
        self._It -= 1
        self._next_time_slice = self._previous_time_slice
        self._previous_time_slice = self._spatial_array_factory(self._data[self._It, ...], self.layout)
        # loading new data invalidates cached delta and output
        self._cached_delta_valid = False
        self._output_valid = False

    def set_time_index(self, It: int) -> None:
        """Set the time index, adjusting the spatial arrays.

        Parameters
        ----------
        It : int
            The desired time index to set.

        Raises
        ------
        IndexError
            If the time index is out of the valid range [0, num_timesteps - 2].
        """
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
        self._previous_time_slice = self._spatial_array_factory(self._data[self._It, ...], self.layout)
        self._next_time_slice = self._spatial_array_factory(self._data[self._It + 1, ...], self.layout)
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
        axes: tuple[ArrayAxis | str, ...],
        staggers: tuple[Stagger | str, ...],
        *,
        attrs: dict[str, Any] | None = None,
    ) -> "TimeDependentField":
        """Create a TimeDependentField from a NumPy array.

        Parameters
        ----------
        data : npt.NDArray
            Input NumPy array containing the field data.
        axes : tuple[ArrayAxis | str, ...]
            Tuple of axes corresponding to the spatial dimensions of the field.
        staggers : tuple[Stagger | str, ...]
            Tuple of staggers corresponding to the spatial dimensions of the field.
        attrs : dict[str, Any] | None, optional
            Attributes for the field.

        Returns
        -------
        TimeDependentField
            A TimeDependentField instance created from the input NumPy array.

        Raises
        ------
        TypeError
            If the input data is not a NumPy array.
        """
        layout = ArrayLayout(axes, staggers)
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected a NumPy array, got {type(data).__name__}")
        return cls(data, layout, NumpyArray, attrs=attrs)

    @classmethod
    def from_dask(
        cls,
        data: da.Array,
        axes: tuple[ArrayAxis | str, ...],
        staggers: tuple[Stagger | str, ...],
        *,
        preload_space: bool = False,
        attrs: dict[str, Any] | None = None,
    ) -> "TimeDependentField":
        """Create a TimeDependentField from a chunked Dask array.

        Parameters
        ----------
        data : da.Array
            Input Dask array containing the field data.
        axes : tuple[ArrayAxis | str, ...]
            Tuple of axes corresponding to the spatial dimensions of the field.
        staggers : tuple[Stagger | str, ...]
            Tuple of staggers corresponding to the spatial dimensions of the field.
        preload_space : bool, optional
            If True, the spatial arrays will be preloaded into memory as NumPy arrays.
            If False (default), the spatial arrays will remain as Dask arrays and will be computed on demand.
        attrs : dict[str, Any] | None, optional
            Attributes for the field.

        Returns
        -------
        TimeDependentField
            A TimeDependentField instance created from the input Dask array.

        Raises
        ------
        TypeError
            If the input data is not a Dask array.
        """
        layout = ArrayLayout(axes, staggers)
        if not isinstance(data, da.Array):
            raise TypeError(f"Expected a Dask array, got {type(data).__name__}")
        if preload_space:
            factory = NumpyArray
        else:
            factory = ChunkedDaskArray
        return cls(data, layout, factory, attrs=attrs)

    @classmethod
    def from_arraylike(
        cls,
        data: npt.ArrayLike,
        axes: tuple[ArrayAxis | str, ...],
        staggers: tuple[Stagger | str, ...],
        *,
        attrs: dict[str, Any] | None = None,
    ) -> "TimeDependentField":
        """Create a TimeDependentField by converting the input to a NumPy array.

        Parameters
        ----------
        data : npt.ArrayLike
            Input data that can be converted to a NumPy array.
        axes : tuple[ArrayAxis | str, ...]
            Tuple of axes corresponding to the spatial dimensions of the field.
        staggers : tuple[Stagger | str, ...]
            Tuple of staggers corresponding to the spatial dimensions of the field.
        attrs : dict[str, Any] | None, optional
            Attributes for the field.

        Returns
        -------
        TimeDependentField
            A TimeDependentField instance created from the input data.

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
        return cls.from_numpy(np.asarray(data), axes, staggers, attrs=attrs)

    @classmethod
    def from_xarray(
        cls,
        data: xr.DataArray,
        time_dim: str,
        dims: Mapping[str, tuple[ArrayAxis | str, Stagger | str]],
        *,
        ignore_missing_dims: bool = False,
    ) -> "TimeDependentField":
        """Create a TimeDependentField from an xarray DataArray.

        Parameters
        ----------
        data : xr.DataArray
            The input xarray DataArray containing the field data.
        time_dim : str
            Name of the time dimension in the DataArray.
        dims : Mapping[str, tuple[ArrayAxis | str, Stagger | str]]
            Mapping of spatial dimension names to ``(ArrayAxis, Stagger)`` tuples.
            Cannot be combined with keyword arguments.
        ignore_missing_dims : bool, optional
            If True, dimensions specified in ``dims`` that are not present in the DataArray will
            be ignored. If False (default), a ValueError will be raised.

        Returns
        -------
        TimeDependentField
            A TimeDependentField instance created from the input xarray DataArray.

        Raises
        ------
        ValueError
            If the time dimension is not found in the dataset dimensions.
            If any of the spatial dimensions are not found in the dataset dimensions.
            If the size of a centered dimension is not provided and the centered dimension is not included in dims.
            If a dimension has a size that does not match the expected size based on the stagger and provided size.

        Notes
        -----
        The spatial dimension names in ``dims`` (or keyword arguments) must exactly
        match all non-time dimensions in ``data`` unless ``ignore_missing_dims`` is True, in which case extra dimensions in ``dims``
        that are not present in ``data`` will be ignored. All non-time dimensions in ``data`` must be accounted for in ``dims``
        regardless of the value of ``ignore_missing_dims``.
        """
        # first ensure time dim exists
        if time_dim not in data.dims:
            raise ValueError(f"Time dimension '{time_dim}' not found in data dimensions {data.dims}")

        # get spatial dims on data
        spatial_dims = [dim for dim in data.dims if dim != time_dim]

        # move time dim to the front if it's not already
        data = data.transpose(time_dim, *spatial_dims)

        # build an array layout from the provided dims
        dims_mapping = dict(dims)  # make a copy to avoid mutating the input
        axes = []
        staggers = []
        for dim in spatial_dims:
            if dim not in dims:
                raise ValueError(f"Dimension '{dim}' in data is missing from dims mapping.")
            axis, stagger = dims_mapping.pop(dim)
            axes.append(ArrayAxis.parse(axis))
            staggers.append(Stagger(stagger))

        # error if there are any extra dimensions in dims that were not in data
        if (not ignore_missing_dims) and dims_mapping:
            raise ValueError(f"Dimensions in dims mapping not found in data: {list(dims_mapping.keys())}")

        array = data.data

        # create field
        if isinstance(array, da.Array):
            return cls.from_dask(
                data=array,
                axes=tuple(axes),
                staggers=tuple(staggers),
                attrs=data.attrs,
            )
        else:
            return cls.from_arraylike(
                data=array,
                axes=tuple(axes),
                staggers=tuple(staggers),
                attrs=data.attrs,
            )


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
