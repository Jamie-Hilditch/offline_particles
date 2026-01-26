"""Submodule for particle kernel launchers."""

import collections
from typing import Callable, TypeVar

import numpy as np

from .fields import FieldData
from .fieldset import Fieldset
from .kernels import BoundKernel, is_active
from .particles import Particles
from .spatial_arrays import BBox

T = TypeVar("T", bound=np.generic)

# Named tuple for time information
Time_info = collections.namedtuple("Time_info", ["time", "tidx", "iteration"])
type Tinfo = Time_info[np.float64 | np.timedelta64, np.float64, int]


class ScalarSource[T]:
    """Descriptor declaring a scalar data source.

    The getter must have signature:
        (self, tinfo: Tinfo) -> np.generic
    """

    def __init__(
        self,
        name: str,
        getter: Callable[[object, Tinfo], T],
    ) -> None:
        self.name = name
        self._getter = getter

    def __get__(self, obj: object | None, owner: type | None = None):
        # Accessed on the class → return descriptor for discovery
        if obj is None:
            return self

        # Accessed on an instance → return bound callable
        def scalar_func(tinfo: Tinfo) -> T:
            return self._getter(obj, tinfo)

        return scalar_func


type ScalarProvider = Callable[[Tinfo], np.generic]

# -------------------------------
# Kernel Launcher
# -------------------------------


class Launcher:
    """Class to launch particle kernels."""

    def __init__(
        self,
        fieldset: Fieldset,
        *,
        index_padding: int = 0,
    ) -> None:
        super().__init__()

        self._scalar_data_sources: dict[str, ScalarProvider] = {}
        self._fieldset = fieldset
        if index_padding < 0:
            raise ValueError("index_padding must be non-negative")
        self._index_padding = index_padding

        # register constants attached to fieldset as scalar data sources
        for name, value in self._fieldset.constants.items():
            value_func = self.create_value_scalar_source(value)
            self.register_scalar_data_source(name, value_func)

    @staticmethod
    def create_value_scalar_source(value: np.generic) -> ScalarProvider:
        """Create a scalar data source that always returns the given value."""

        def value_func(tinfo: Tinfo) -> np.generic:
            return value

        return value_func

    def register_scalar_data_source(self, name: str, source: ScalarProvider) -> None:
        """Register a scalar data source function."""
        if name in self._scalar_data_sources:
            raise ValueError(
                f"Scalar data source '{name}' is already registered. Deregister it before registering a new one."
            )
        if name in self._fieldset.fields:
            raise ValueError(f"Scalar data source '{name}' conflicts with a field in the fieldset.")

        self._scalar_data_sources[name] = source

    def deregister_scalar_data_source(self, name: str) -> None:
        """Deregister a scalar data source function."""
        if name not in self._scalar_data_sources:
            raise ValueError(f"Scalar data source '{name}' is not registered.")
        del self._scalar_data_sources[name]

    def register_scalar_data_sources_from_object(self, obj: object):
        """Scan object for scalar data source functions and register them."""
        for attr_name in dir(type(obj)):
            attr = getattr(type(obj), attr_name)
            if isinstance(attr, ScalarSource):
                scalar_func = getattr(obj, attr_name)
                self.register_scalar_data_source(attr.name, scalar_func)

    @property
    def index_padding(self) -> int:
        """The index padding used by this launcher."""
        return self._index_padding

    def set_index_padding(self, index_padding: int, force: bool = False) -> None:
        """Set the index padding using by this launcher.

        Unless `force` is True, only increases the index padding.
        """
        if index_padding < 0:
            raise ValueError("Index padding must be non-negative.")
        if force or index_padding > self._index_padding:
            self._index_padding = index_padding

    def construct_bbox(
        self,
        particles: Particles,
    ) -> BBox:
        """Construct a bounding box around the given particles with index padding."""
        idx = is_active(particles["status"])

        z_indices = particles["zidx"][idx]
        y_indices = particles["yidx"][idx]
        x_indices = particles["xidx"][idx]

        zmin = z_indices.min(initial=np.inf) - self._index_padding
        zmax = z_indices.max(initial=-np.inf) + self._index_padding

        ymin = y_indices.min(initial=np.inf) - self._index_padding
        ymax = y_indices.max(initial=-np.inf) + self._index_padding

        xmin = x_indices.min(initial=np.inf) - self._index_padding
        xmax = x_indices.max(initial=-np.inf) + self._index_padding

        return BBox(
            zmin=zmin,
            zmax=zmax,
            ymin=ymin,
            ymax=ymax,
            xmin=xmin,
            xmax=xmax,
        )

    def get_field_data(self, name: str, time_index: float, bbox: BBox) -> FieldData:
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
            Tuple containing the field data array and offsets.
        """
        return self._fieldset[name].get_field_data(time_index, bbox)

    def launch_kernel(self, bound_kernel: BoundKernel, particles: Particles, tinfo: Tinfo) -> None:
        """Launch a kernel."""
        # construct kernel inputs
        bbox = self.construct_bbox(particles)
        particle_properties = {
            name: particles[binding] for name, binding in bound_kernel.particle_property_bindings.items()
        }
        scalars = {
            name: self._scalar_data_sources[binding](tinfo) for name, binding in bound_kernel.scalars_bindings.items()
        }
        field_data = {
            name: self.get_field_data(binding, tinfo.tidx, bbox)
            for name, binding in bound_kernel.field_data_bindings.items()
        }
        # call the kernel
        bound_kernel.kernel(particle_properties, scalars, field_data)
