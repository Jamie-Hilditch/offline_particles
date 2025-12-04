"""Submodule for constructing and running kernels on particles."""

from typing import Callable, Iterable, Self

import numba
import numpy as np
import numpy.typing as npt

from .kernel_data import KernelData

type Particle = npt.NDArray
type KernelFunction = Callable[[Particle, KernelData, ...], None]


class ParticleKernel:
    """A kernel to be executed on a particle."""

    def __init__(
        self,
        kernel_function: KernelFunction,
        particle_fields: dict[str, npt.DTypeLike],
        simulation_fields: Iterable[str],
    ) -> None:
        self._kernel_function = numba.njit(nogil=True, fastmath=True)(kernel_function)
        self._particle_fields = {
            field: np.dtype(dtype) for field, dtype in particle_fields.items()
        }
        self._simulation_fields = tuple(simulation_fields)
        self._vector_kernel_function = _vectorize_kernel_function(self._kernel_function)

    @property
    def particle_fields(self) -> dict[str, np.dtype]:
        """The particle fields required by this kernel."""
        return self._particle_fields

    @property
    def simulation_fields(self) -> tuple[str]:
        """The simulation fields required by this kernel."""
        return self._simulation_fields

    def chain_with(self, other: Self) -> Self:
        """Create a ParticleKernel by chaining this kernel with another."""

        combined_particle_fields = merge_particle_fields(
            self.particle_fields, other.particle_fields
        )
        combined_simulation_fields = tuple(
            set(self.simulation_fields).union(other.simulation_fields)
        )

        # find the indices of the fields in the combined tuple
        # these are tuples of ints so numba will treat them as compile time constants
        first_indices = tuple(
            combined_simulation_fields.index(f) for f in self.simulation_fields
        )
        second_indices = tuple(
            combined_simulation_fields.index(f) for f in other.simulation_fields
        )

        first_function = self._kernel_function
        second_function = other._kernel_function

        chained_kernel = _chain_kernel_functions(
            first_function,
            second_function,
            first_indices,
            second_indices,
        )

        return ParticleKernel(
            chained_kernel, combined_particle_fields, combined_simulation_fields
        )

    @classmethod
    def from_sequence(cls, kernels: Iterable["ParticleKernel"]) -> "ParticleKernel":
        """Create a ParticleKernel by combining a sequence of ParticleKernels."""
        kernel_iter = iter(kernels)
        combined_kernel = next(kernel_iter)
        for kernel in kernel_iter:
            combined_kernel = combined_kernel.chain_with(kernel)
        return combined_kernel


def merge_particle_fields(kernels: Iterable[ParticleKernel]) -> dict[str, np.dtype]:
    """Merge the particle fields required by a sequence of ParticleKernels."""
    merged_fields: dict[str, npt.DType] = {}
    for kernel in kernels:
        for field, dtype in kernel.particle_fields.items():
            if field in merged_fields:
                if merged_fields[field] != np.dtype(dtype):
                    raise ValueError(
                        f"Conflicting dtypes for particle field '{field}': "
                        f"{merged_fields[field]} vs {dtype}"
                    )
            else:
                merged_fields[field] = np.dtype(dtype)


def _vectorize_kernel_function(
    particle_kernel_function: KernelFunction,
) -> KernelFunction:
    """Create a vectorized version of a particle kernel function."""

    @numba.njit(nogil=True, fastmath=True, parallel=True)
    def vectorized_kernel_function(
        particles: Particle, *kernel_data: KernelData
    ) -> None:
        n_particles = particles.shape[0]
        for i in numba.prange(n_particles):
            particle_kernel_function(particles[i], *kernel_data)

    return vectorized_kernel_function


def _chain_kernel_functions(
    first_function: KernelFunction,
    second_function: KernelFunction,
    first_indices: tuple[int, ...],
    second_indices: tuple[int, ...],
) -> KernelFunction:
    """Chain two kernel functions together."""

    def chained_function(p: Particle, *kernel_data: KernelData) -> None:
        first_kernel_data = tuple(kernel_data[i] for i in first_indices)
        second_kernel_data = tuple(kernel_data[i] for i in second_indices)

        first_function(p, *first_kernel_data)
        second_function(p, *second_kernel_data)

    return chained_function
