"""Particle Kernels."""

import types
from typing import Callable, Iterable, Mapping, NamedTuple, Self

import numpy as np
import numpy.typing as npt

from ..fields import FieldData

type KernelFunction = Callable[[NamedTuple, dict[str, npt.NDArray], dict[str, FieldData]], None]

DEFAULT_PARTICLE_FIELDS: dict[str, npt.DTypeLike] = {
    "status": np.uint8,
    "zidx": np.float64,
    "yidx": np.float64,
    "xidx": np.float64,
}

class ParticleKernel:
    """A kernel to be execute on particles."""

    def __init__(
        self, 
        fn: KernelFunction | Iterable[KernelFunction], 
        particle_fields: dict[str, npt.DTypeLike],
        scalars: dict[str, npt.DTypeLike],
        simulation_fields: Iterable[str],
    ):
        # store kernels as tuple of functions
        if callable(fn):
            self._funcs = (fn,)
        else:
            funcs = tuple(fn)
            if not all(callable(f) for f in funcs):
                raise TypeError("All kernel functions must be callable")
            self._funcs = funcs
        
        self._particle_fields = {
            field: np.dtype(dtype) for field, dtype in particle_fields.items()
        }
        self._scalars = {
            scalar: np.dtype(dtype) for scalar, dtype in scalars.items()
        }
        self._simulation_fields = frozenset(simulation_fields)

    @property
    def particle_fields(self) -> Mapping[str, np.dtype]:
        """The required particle fields and their dtypes."""
        return types.MappingProxyType(self._particle_fields)

    @property
    def scalars(self) -> Mapping[str, np.dtype]:
        """The required scalar fields and their dtypes."""
        return types.MappingProxyType(self._scalars)

    @property
    def simulation_fields(self) -> frozenset[str]:
        """The required simulation fields."""
        return self._simulation_fields

    @property
    def functions(self) -> tuple[KernelFunction, ...]:
        """The kernel functions."""
        return self._funcs

    @staticmethod
    def func_name(fn: KernelFunction) -> str:
        return getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))

    def __repr__(self) -> str:
        funcs = ", ".join(self.func_name(fn) for fn in self._funcs)

        return (
            f"{self.__class__.__name__}("
            f"funcs=[{funcs}], "
            f"particle_fields={list(self._particle_fields)}, "
            f"scalars={list(self._scalars)}, "
            f"simulation_fields={sorted(self._simulation_fields)}"
            f")"
        )

    def __str__(self) -> str:
        return "Particle Kernel: " + " → ".join(self.func_name(fn) for fn in self._funcs)


    def __call__(
        self, 
        particles: NamedTuple, 
        scalars: dict[str, npt.NDArray], 
        fielddata: dict[str, FieldData]
    ) -> None:
        """Execute the kernel on the given particles."""
        for fn in self._funcs:
            fn(particles, scalars, fielddata)

    @classmethod 
    def chain(
        cls,
        *kernels: Self
    ) -> Self:
        """Create a ParticleKernel by merging multiple kernels."""
        funcs = tuple(fn for kernel in kernels for fn in kernel._funcs)
        particle_fields = _merge_particle_fields(kernels)
        scalars = _merge_scalars(kernels)
        simulation_fields = _merge_simulation_fields(kernels)

        return cls(
            funcs,
            particle_fields,
            scalars,
            simulation_fields,
        )

    def chain_with(self, *others: Self) -> Self:
        """Chain this kernel with other kernels."""
        return ParticleKernel.chain(self, *others)

def _merge_particle_fields(
    kernels: Iterable[ParticleKernel]
) -> dict[str, np.dtype]:
    """Merge particle fields from multiple kernels."""
    merged_fields: dict[str, np.dtype] = DEFAULT_PARTICLE_FIELDS.copy()
    for kernel in kernels:
        for field, dtype in kernel.particle_fields.items():
            if field in merged_fields:
                if merged_fields[field] != dtype:
                    raise TypeError(
                        f"Conflicting dtypes for particle field '{field}': "
                        f"{merged_fields[field]} vs {dtype}"
                    )
            else:
                merged_fields[field] = dtype
    return merged_fields

def _merge_scalars(
    kernels: Iterable[ParticleKernel]
) -> dict[str, np.dtype]:
    """Merge scalar fields from multiple kernels."""
    merged_scalars: dict[str, np.dtype] = {}
    for kernel in kernels:
        for scalar, dtype in kernel.scalars.items():
            if scalar in merged_scalars:
                if merged_scalars[scalar] != dtype:
                    raise TypeError(
                        f"Conflicting dtypes for scalar '{scalar}': "
                        f"{merged_scalars[scalar]} vs {dtype}"
                    )
            else:
                merged_scalars[scalar] = dtype

    return merged_scalars

def _merge_simulation_fields(
    kernels: Iterable[ParticleKernel]
) -> frozenset[str]:
    """Merge simulation fields from multiple kernels."""
    merged_fields: set[str] = set()
    for kernel in kernels:
        merged_fields.update(kernel.simulation_fields)
    return frozenset(merged_fields)