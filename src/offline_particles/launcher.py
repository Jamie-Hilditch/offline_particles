"""Submodule for particle kernel launchers."""

from numbers import Number
from typing import Callable

import numpy.typing as npt

from .fields import FieldData
from .fieldset import Fieldset
from .particle_kernel import ParticleKernel
from .spatial_arrays import BBox

type ScalarDataSource = Callable[[float], Number]

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

        self._scalar_data_sources: dict[str, ScalarDataSource] = {}
        self._fieldset = fieldset
        if index_padding < 0:
            raise ValueError("index_padding must be non-negative")
        self._index_padding = index_padding

        # register constants attached to fieldset as scalar data sources
        for name, value in self._fieldset.constants.items():
            self.register_scalar_data_source(
                name, lambda time_index: value
            )

    def register_scalar_data_source(self, name: str, func: ScalarDataSource) -> None:
        """Register a scalar data source function."""
        if name in self._scalar_data_sources:
            raise ValueError(
                f"Scalar data source '{name}' is already registered. Deregister it before registering a new one."
            )
        if name in self._fieldset.fields:
            raise ValueError(
                f"Scalar data source '{name}' conflicts with a field in the fieldset."
            )
        self._scalar_data_sources[name] = func

    def deregister_scalar_data_source(self, name: str) -> None:
        """Deregister a scalar data source function."""
        if name not in self._scalar_data_sources:
            raise ValueError(f"Scalar data source '{name}' is not registered.")
        del self._scalar_data_sources[name]

    def register_scalar_data_sources_from_object(self, obj):
        """Scan object for scalar data source functions and register them."""
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name)
            if callable(attr) and hasattr(attr, "__scalar_data_name__"):
                name = getattr(attr, "__scalar_data_name__")
                self.register_scalar_data_source(name, attr)

    @property
    def index_padding(self) -> int:
        """The index padding used by this launcher."""
        return self._index_padding

    def maybe_increase_index_padding(self, padding: int) -> None:
        """Increase the index padding if the given padding is larger.

        Parameters
        ----------
        padding : int
            The padding to compare against the current index padding.
        """
        if padding > self._index_padding:
            self._index_padding = padding

    def construct_bbox(
        self,
        particles: npt.NDArray,
    ) -> BBox:
        """Construct a bounding box around the given particles with index padding."""

        z_indices = particles["zidx"]
        y_indices = particles["yidx"]
        x_indices = particles["xidx"]

        zmin = z_indices.min() - self._index_padding
        zmax = z_indices.max() + self._index_padding

        ymin = y_indices.min() - self._index_padding
        ymax = y_indices.max() + self._index_padding

        xmin = x_indices.min() - self._index_padding
        xmax = x_indices.max() + self._index_padding

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

    def launch_kernel(
        self, kernel: ParticleKernel, particles: npt.NDArray, time_index: float
    ) -> None:
        """Launch a kernel."""
        kernel_arguments = []
        # scalars go first
        for name in kernel.scalars:
            kernel_arguments.append(
                self._scalar_data_sources[name](time_index)
            )
        # then fields
        bbox = self.construct_bbox(particles)
        for name in kernel.simulation_fields:
            array, offsets = self.get_field_data(name, time_index, bbox)
            kernel_arguments.append(array)
            kernel_arguments.append(offsets)
       
        # call the vectorized kernel function
        kernel._vector_kernel_function(particles, *kernel_arguments)


def register_scalar_data_source(
    name: str,
) -> Callable[[ScalarDataSource], ScalarDataSource]:
    """A decorator to register a scalar data source.

    Parameters
    ----------
    name : str
        Name to register the function under.
    """

    def decorator(func: ScalarDataSource) -> ScalarDataSource:
        func.__scalar_data_name__ = name
        return func

    return decorator
