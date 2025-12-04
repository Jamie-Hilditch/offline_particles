"""Submodule for particle kernel launchers."""


import numpy.typing as npt

from .kernel_data import KernelData, KernelDataFunction, KernelDataSource
from .particle_kernel import ParticleKernel
from .spatial_arrays import BBox

# -------------------------------
# Kernel Launcher
# -------------------------------


class Launcher:
    """Class to launch particle kernels."""

    def __init__(
        self,
        index_padding: int = 0,
    ) -> None:
        super().__init__()

        if index_padding < 0:
            raise ValueError("index_padding must be non-negative")
        self._index_padding = index_padding
        self._kernel_data_sources: dict[str, KernelDataFunction] = {}

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

    def register_kernel_data_functions_from_source(self, source: KernelDataSource):
        """
        Register all kernel-data providers from any KernelDataSource object.

        Parameters
        ----------
        source : KernelDataSource
            An instance (or subclass instance) whose kernel-data
            functions will be added to this launcher.
        """

        if not isinstance(source, KernelDataSource):
            raise TypeError(
                f"Cannot register object of type {type(source)}; "
                f"it must be a KernelDataSource."
            )

        # Iterate through the source's registered providers
        for name, func in source.kernel_data_items():
            if name in self._kernel_data_sources:
                raise ValueError(
                    f"Kernel data source '{name}' already registered in the launcher."
                )

            # Register the bound method *from the source*, not from the class.
            self._kernel_data_sources[name] = func

    def deregister_kernel_data_function(self, name: str) -> None:
        """
        Deregister a kernel-data provider by name.

        Parameters
        ----------
        name : str
            Name of the kernel-data provider to deregister.
        """

        if name not in self._kernel_data_sources:
            raise ValueError(f"Kernel data source '{name}' not registered in launcher.")

        del self._kernel_data_sources[name]

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

    def get_kernel_data(self, name: str, time_index: float, bbox: BBox) -> KernelData:
        """Get the field data at a given time index.

        Parameters
        ----------
        time_index : float
            Time index.
        bbox : BBox
            Bounding box to extract data from defined in terms of centered grid indices.

        Returns
        -------
        KernelData
            Namedtuple containing the field data array, the dimension mask, and offsets.
        """
        kernel_data_function = self._kernel_data_sources.get(name, None)
        if kernel_data_function is None:
            raise ValueError(f"Field {name} not registered with launcher.")
        return kernel_data_function(time_index, bbox)

    def launch_kernel(
        self, kernel: ParticleKernel, particles: npt.NDArray, time_index: float
    ) -> None:
        """Launch a kernel."""
        # Construct the bounding box around the particles.
        bbox = self.construct_bbox(particles)

        # gather the field data required by the kernel
        kernel_data = []
        for name in kernel.simulation_fields:
            kernel_data.append(self.get_kernel_data(name, time_index, bbox))

        # call the vectorized kernel function
        kernel._vector_kernel_function(particles, *kernel_data)
