"""Submodule for constructing and running kernels on particles."""

from typing import Callable, Iterable, Self

import numba
import numpy as np
import numpy.typing as npt

type Particle = npt.NDArray
type KernelFunction = Callable[[Particle, ...], None]

DEFAULT_PARTICLE_FIELDS: dict[str, npt.DTypeLike] = {
    "status": np.uint8,
    "zidx": np.float64,
    "yidx": np.float64,
    "xidx": np.float64,
}

class ParticleKernel:
    """A kernel to be executed on a particle."""

    def __init__(
        self,
        kernel_function: KernelFunction,
        particle_fields: dict[str, npt.DTypeLike],
        scalars: Iterable[str],
        simulation_fields: Iterable[str],
        *,
        fastmath: bool = True,
        nogil: bool = True,
        parallel: bool = True,
    ) -> None:
        self._kernel_function = numba.njit(nogil=True, fastmath=True)(kernel_function)
        self._particle_fields = {
            field: np.dtype(dtype) for field, dtype in particle_fields.items()
        }
        self._scalars = tuple(scalars)
        self._simulation_fields = tuple(simulation_fields)
        self._fastmath = fastmath
        self._nogil = nogil
        self._parallel = parallel
        self._vector_kernel_function = _vectorize_kernel_function(
            self._kernel_function, fastmath=fastmath, nogil=nogil, parallel=parallel
        )

    @property
    def particle_fields(self) -> dict[str, np.dtype]:
        """The particle fields required by this kernel."""
        return self._particle_fields

    @property
    def scalars(self) -> tuple[str]:
        """The scalars required by this kernel."""
        return self._scalars

    @property
    def simulation_fields(self) -> tuple[str]:
        """The simulation fields required by this kernel."""
        return self._simulation_fields

    @property
    def fastmath(self) -> bool:
        """Whether the kernel function is compiled with fastmath."""
        return self._fastmath

    @property
    def nogil(self) -> bool:
        """Whether the kernel function is compiled with nogil."""
        return self._nogil

    @property
    def parallel(self) -> bool:
        """Whether the kernel function is compiled with parallel."""
        return self._parallel

    def chain_with(self, other: Self) -> Self:
        """Create a ParticleKernel by chaining this kernel with another."""

        combined_particle_fields = merge_particle_fields(
            self.particle_fields, other.particle_fields
        )
        combined_scalars = tuple(set(self.scalars).union(other.scalars))
        combined_simulation_fields = tuple(
            set(self.simulation_fields).union(other.simulation_fields)
        )

        nscalars = len(combined_scalars)
        # find indices of the arguments in the combined argument list
        first_indices = []
        second_indices = []

        # first the scalars
        for s in self.scalars:
            first_indices.append(combined_scalars.index(s))
        for s in other.scalars:
            second_indices.append(combined_scalars.index(s))

        # then the fields
        for f in self.simulation_fields:
            first_indices.append(nscalars + 2 * combined_simulation_fields.index(f))
            first_indices.append(nscalars + 2 * combined_simulation_fields.index(f) + 1)
        tuple(combined_scalars.index(s) for s in self.scalars)
        for f in other.simulation_fields:
            second_indices.append(nscalars + 2 * combined_simulation_fields.index(f))
            second_indices.append(
                nscalars + 2 * combined_simulation_fields.index(f) + 1
            )

        first_function = self._kernel_function
        second_function = other._kernel_function

        chained_kernel = _chain_kernel_functions(
            first_function,
            second_function,
            first_indices,
            second_indices,
        )

        # compilation options
        fastmath = self.fastmath and other.fastmath
        nogil = self.nogil and other.nogil
        parallel = self.parallel and other.parallel

        return ParticleKernel(
            chained_kernel,
            combined_particle_fields,
            combined_scalars,
            combined_simulation_fields,
            fastmath=fastmath,
            nogil=nogil,
            parallel=parallel,
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
    merged_fields = DEFAULT_PARTICLE_FIELDS.copy()
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
    return merged_fields


def _vectorize_kernel_function(
    particle_kernel_function: KernelFunction,
    fastmath: bool,
    nogil: bool,
    parallel: bool,
) -> KernelFunction:
    """Create a vectorized version of a particle kernel function."""

    @numba.njit(nogil=nogil, fastmath=fastmath, parallel=parallel)
    def vectorized_kernel_function(particles: Particle, *kernel_data) -> None:
        n_particles = particles.shape[0]
        for i in numba.prange(n_particles):
            # skip inactive particles
            if particles["status"][i] > 0:
                continue
            particle_kernel_function(particles[i], *kernel_data)

    return vectorized_kernel_function


def _chain_kernel_functions(
    first_function: KernelFunction,
    second_function: KernelFunction,
    first_indices: Iterable[int],
    second_indices: Iterable[int],
) -> KernelFunction:
    """Build a chained kernel function by statically generating the source code."""

    # argument list for the first and second functions
    a_args = ", ".join(f"args[{i}]" for i in first_indices)
    b_args = ", ".join(f"args[{i}]" for i in second_indices)

    # source code
    src = []
    src.append("def chained_function(p, *args):")
    src.append(f"    first_function(p, {a_args})")
    src.append(f"    second_function(p, {b_args})")
    src_code = "\n".join(src)

    # local namespace for exec
    local_ns = {
        "first_function": first_function,
        "second_function": second_function,
    }
    exec(src_code, local_ns)
    return local_ns["chained_function"]
