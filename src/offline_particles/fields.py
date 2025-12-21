"""Submodule for handling fields in ROMS particle tracking simulations."""

import abc
import collections
from typing import Callable

import dask.array as da
import numpy as np
import numpy.typing as npt

from .spatial_arrays import BBox, ChunkedDaskArray, NumpyArray, SpatialArray, Stagger

FieldData = collections.namedtuple("FieldData", ["array", "offsets"])


class Field(abc.ABC):
    """Abstract base class for fields used in particle tracking."""

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
        self._dmask = tuple(
            map(int, (z_stagger.is_active, y_stagger.is_active, x_stagger.is_active))
        )

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
    def dmask(self) -> tuple[int, int, int]:
        """Dimension mask indicating active dimensions."""
        return self._dmask

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
    ):
        super().__init__(
            z_stagger=data.z_stagger,
            y_stagger=data.y_stagger,
            x_stagger=data.x_stagger,
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
            raise ValueError(
                f"Expected shape {expected_shape} but data has shape {self._data.shape}"
            )

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
    ) -> "StaticField":
        """Create a StaticField from a NumPy array."""
        spatial_array = NumpyArray(
            data=data,
            z_stagger=Stagger(z_stagger),
            y_stagger=Stagger(y_stagger),
            x_stagger=Stagger(x_stagger),
        )
        return cls(data=spatial_array)

    @classmethod
    def from_dask(
        cls,
        data: da.Array,
        z_stagger: Stagger | str,
        y_stagger: Stagger | str,
        x_stagger: Stagger | str,
    ) -> "StaticField":
        """Create a StaticField from a chunked Dask array."""
        spatial_array = ChunkedDaskArray(
            data=data,
            z_stagger=Stagger(z_stagger),
            y_stagger=Stagger(y_stagger),
            x_stagger=Stagger(x_stagger),
        )
        return cls(data=spatial_array)


type SpatialArrayFactory = Callable[
    [da.Array | npt.NDArray, Stagger, Stagger, Stagger], SpatialArray
]


class TimeDependentField(Field):
    """Class representing a time-dependent field with an least 1 spatial dimension."""

    def __init__(
        self,
        data: da.Array | npt.NDArray,
        z_stagger: Stagger | str,
        y_stagger: Stagger | str,
        x_stagger: Stagger | str,
        spatial_array_factory: SpatialArrayFactory = NumpyArray,
        output_dtype: npt.DTypeLike = np.float64,
    ):
        super().__init__(
            z_stagger=Stagger(z_stagger),
            y_stagger=Stagger(y_stagger),
            x_stagger=Stagger(x_stagger),
        )

        if data.ndim < 2:
            raise ValueError(
                "TimeDependentField requires at least 2 dimensions (time + spatial). For spatially invariant fields, use ConstantField or TemporalField."
            )
        self._data = data
        self._spatial_array_factory = spatial_array_factory
        self._data_dtype = data.dtype
        self._output_dtype = output_dtype

        # temporary arrays for interpolation 
        self._allocate_interpolation_arrays((0,) * (data.ndim - 1))

        # time index
        if self._data.shape[0] < 2:
            raise ValueError("TimeDependentField requires at least 2 time steps.")
        self._num_timesteps = self._data.shape[0]
        self._It = 0
        self._current_time_slice = self._spatial_array_factory(
            self._data[0, ...], self.z_stagger, self.y_stagger, self.x_stagger
        )
        self._next_time_slice = self._spatial_array_factory(
            self._data[1, ...], self.z_stagger, self.y_stagger, self.x_stagger
        )

    def _allocate_interpolation_arrays(self, shape: tuple[int, ...]) -> None:
        """Allocate temporary arrays for interpolation."""
        self._array_shape = shape
        self._gt_current = np.empty(shape=shape, dtype=self._data.dtype)
        self._ft_next = np.empty(shape=shape, dtype=self._data.dtype)
        self._output = np.empty(shape=shape, dtype=self._output_dtype)

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
            raise ValueError(
                f"Expected shape {expected_shape} but data has shape {self._data.shape}"
            )

    @property
    def current_time_slice(self) -> SpatialArray:
        """Get the current time slice as a SpatialArray."""
        return self._current_time_slice

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
        self._current_time_slice = self._next_time_slice
        self._next_time_slice = self._spatial_array_factory(
            self._data[self._It + 1, ...],
            self.z_stagger,
            self.y_stagger,
            self.x_stagger,
        )

    def set_time_index(self, It: int) -> None:
        """Set the time index, adjusting the spatial arrays."""
        # if current time index do nothing
        if It == self._It:
            return
        # if it's the next timestep we can increment
        if It == self._It + 1:
            return self.increment_time()
        # else check range
        if It < 0 or It > self._num_timesteps - 2:
            raise IndexError(
                f"Valid range of time indices is 0,...,{self._num_timesteps - 2}, got {It}."
            )

        self._It = It
        self._current_time_slice = self._spatial_array_factory(
            self._data[self._It, ...], self.z_stagger, self.y_stagger, self.x_stagger
        )
        self._next_time_slice = self._spatial_array_factory(
            self._data[self._It + 1, ...],
            self.z_stagger,
            self.y_stagger,
            self.x_stagger,
        )

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

        # load the two time subsets
        current_data, offsets = self._current_time_slice.get_data_subset(bbox)
        next_data, _ = self._next_time_slice.get_data_subset(bbox)

        # linear interpolation in time
        if self._array_shape != current_data.shape:
            self._allocate_interpolation_arrays(current_data.shape)
        
        ft = self._data_dtype.type(ft) 
        gt = self._data_dtype.type(1 - ft)

        np.multiply(current_data, gt, out=self._gt_current)
        np.multiply(next_data, ft, out=self._ft_next)
        np.add(self._gt_current, self._ft_next, out=self._output, casting='unsafe')

        return FieldData(self._output, offsets)

    @classmethod
    def from_numpy(
        cls,
        data: npt.NDArray,
        z_stagger: Stagger | str,
        y_stagger: Stagger | str,
        x_stagger: Stagger | str,
    ) -> "TimeDependentField":
        """Create a TimeDependentField from a NumPy array."""
        return cls(data, z_stagger, y_stagger, x_stagger, NumpyArray)

    @classmethod
    def from_dask(
        cls,
        data: da.Array,
        z_stagger: Stagger | str,
        y_stagger: Stagger | str,
        x_stagger: Stagger | str,
        *,
        preload_space: bool = False,
    ) -> "TimeDependentField":
        """Create a TimeDependentField from a chunked Dask array."""
        if preload_space:
            factory = NumpyArray
        else:
            factory = ChunkedDaskArray
        return cls(data, z_stagger, y_stagger, x_stagger, factory)
