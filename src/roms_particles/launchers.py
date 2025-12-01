"""Submodule for particle kernel launchers."""

import abc

import numba
import numpy.typing as npt

from .fieldset import Fieldset
from .kernel import ParticleKernel, FieldData
from .spatial_arrays import BBox

from typing import Iterable, Callable


class Launcher(abc.ABC):
    """Class to launch particle kernels with required fields."""

    launcher_fields: tuple[str, ...] = ()

    def __init__(
        self,
        fieldset: Fieldset,
        index_padding: int = 0,
    ) -> None:
        self._fieldset = fieldset
        if index_padding < 0:
            raise ValueError("index_padding must be non-negative")
        self._index_padding = index_padding

    @property
    def fieldset(self) -> Fieldset:
        """The Fieldset used by this launcher."""
        return self._fieldset

    def index_padding(self) -> int:
        """The index padding used by this launcher."""
        return self._index_padding

    def construct_bbox(
        self,
        particles: npt.NDArray,
    ) -> BBox:
        """Construct a bounding box around the given particles with index padding."""

        z_indices = particles['z_idx']
        y_indices = particles['y_idx']
        x_indices = particles['x_idx']

        z_min = z_indices.min() - self._index_padding
        z_max = z_indices.max() + self._index_padding

        y_min = y_indices.min() - self._index_padding
        y_max = y_indices.max() + self._index_padding

        x_min = x_indices.min() - self._index_padding
        x_max = x_indices.max() + self._index_padding

        return BBox(
            z_min=z_min,
            z_max=z_max,
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
        )

    @abc.abstractmethod
    def get_field_data(self, name: float, time_index: float, bbox: BBox) -> FieldData:
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
            Namedtuple containing the field data array, the dimension mask, and offsets.
        """
        pass

    @abc.abstractmethod
    def kernels(self) -> Iterable[ParticleKernel]:
        """All the kernels to be launched."""
        pass

    def launch_kernel(self, kernel: ParticleKernel, time_index: float, particles: npt.NDArray) -> None:
        """Launch a single kernel."""

        # prepare field data
        bbox = self.construct_bbox(particles)
        field_data = []
        for name in kernel.simulation_fields:
            if name in self.launcher_fields:
                field_data.append(self.get_field_data(name, time_index, bbox))
            elif name in self.fieldset:
                field_data.append(self.fieldset.get_field_data(name, time_index, bbox))
            else:
                raise ValueError(f"Field {name} required by kernel not found in fieldset.")

        # call the vectorized kernel function
        kernel._vector_kernel_function(
            particles,
            *field_data
        )

                
