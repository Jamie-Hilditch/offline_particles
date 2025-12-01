"""Submodule for constructing and running kernels on particles."""

import collections
import numba 
import numpy.typing as npt

from typing import Callable, Iterable, Self

"""The FieldData type for passing data into kernels."""
FieldData = collections.namedtuple("FieldData", ("array", "dmask", "offsets"))

type Particle = npt.NDArray
type KernelFunction = Callable[[Particle, FieldData, ...], None]

class ParticleKernel:
    """A kernel to be executed on a particle."""

    def __init__(self, 
        kernel_function: KernelFunction,
        particle_fields: Iterable[str],
        simulation_fields: Iterable[str],
    ) -> None:
        self._kernel_function = numba.njit(nogil=True, fastmath=True)(kernel_function)
        self._particle_fields = set(particle_fields)
        self._simulation_fields = tuple(simulation_fields)
        self._vector_kernel_function = _vectorize_kernel_function(self._kernel_function)

    @property
    def particle_fields(self) -> set[str]:
        """The particle fields required by this kernel."""
        return self._particle_fields

    @property
    def simulation_fields(self) -> tuple[str]:
        """The simulation fields required by this kernel."""
        return self._simulation_fields

    def chain_with(self, other: Self) -> Self:
        """Create a ParticleKernel by chaining this kernel with another."""

        combined_particle_fields = set(self.particle_fields).union(other.particle_fields)
        combined_simulation_fields = tuple(set(self.simulation_fields).union(other.simulation_fields))
        
        # find the indices of the fields in the combined tuple
        # these are tuples of ints so numba will treat them as compile time constants
        first_indices = tuple(combined_simulation_fields.index(f) for f in self.simulation_fields)
        second_indices = tuple(combined_simulation_fields.index(f) for f in other.simulation_fields)

        first_function = self._kernel_function
        second_function = other._kernel_function

        chained_kernel = _chain_kernel_functions(
            first_function,
            second_function,
            first_indices,
            second_indices,
        )

        return ParticleKernel(
            chained_kernel,
            combined_particle_fields,
            combined_simulation_fields
        )

    @classmethod
    def from_sequence(
        cls, 
        kernels: Iterable["ParticleKernel"]
    ) -> "ParticleKernel":
        """Create a ParticleKernel by combining a sequence of ParticleKernels."""
        kernel_iter = iter(kernels)
        combined_kernel = next(kernel_iter)
        for kernel in kernel_iter:
            combined_kernel = combined_kernel.chain_with(kernel)
        return combined_kernel


def _vectorize_kernel_function(
    particle_kernel_function: KernelFunction
) -> KernelFunction:
    """Create a vectorized version of a particle kernel function."""

    @numba.njit(nogil=True, fastmath=True, parallel=True)
    def vectorized_kernel_function(
        particles: Particle,
        *field_data: FieldData
    ) -> None:
        n_particles = particles.shape[0]
        for i in numba.prange(n_particles):
            particle_kernel_function(
                particles[i],
                *field_data
            )

    return vectorized_kernel_function

def _chain_kernel_functions(
    first_function: KernelFunction,
    second_function: KernelFunction,
    first_indices: tuple[int, ...],
    second_indices: tuple[int, ...],
) -> KernelFunction:
    """Chain two kernel functions together."""

    def chained_function(
        p: Particle,
        *field_data: FieldData
    ) -> None:
        first_field_data = tuple(field_data[i] for i in first_indices)
        second_field_data = tuple(field_data[i] for i in second_indices)

        first_function(
            p, 
            *first_field_data
        )
        second_function(
            p,  
            *second_field_data
        )

    return chained_function