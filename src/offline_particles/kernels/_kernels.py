"""Particle Kernels."""

from __future__ import annotations

import dataclasses
import types
from typing import Callable, Iterable, Mapping, Self

import numpy as np
import numpy.typing as npt

from ..fields import Field, FieldData
from ..spatial_arrays import ALL_STAGGERS, Stagger

type ParticlePropertiesType = Mapping[str, npt.NDArray]
type ScalarsType = Mapping[str, np.generic]
type FieldDataType = Mapping[str, FieldData]
type KernelFunction = Callable[[ParticlePropertiesType, ScalarsType, FieldDataType], None]


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class KernelInputDeclaration:
    """Declaration of a kernel input."""

    name: str
    dtype: np.dtype[np.generic]

    def __init__(self, name: str, dtype: npt.DTypeLike) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", np.dtype(dtype))

    @property
    def doc_string_part(self) -> str:
        return f"'{self.name}' ({self.dtype})"


class ParticlePropertyDeclaration(KernelInputDeclaration):
    """Declaration of a particle property required by a kernel."""


class ScalarDeclaration(KernelInputDeclaration):
    """Declaration of a scalar required by a kernel."""


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class FieldDataDeclaration(KernelInputDeclaration):
    """Declaration of field data required by a kernel."""

    z_staggers: frozenset[Stagger]
    y_staggers: frozenset[Stagger]
    x_staggers: frozenset[Stagger]

    def __init__(
        self,
        name: str,
        dtype: npt.DTypeLike,
        *,
        z_staggers: Iterable[Stagger] = ALL_STAGGERS,
        y_staggers: Iterable[Stagger] = ALL_STAGGERS,
        x_staggers: Iterable[Stagger] = ALL_STAGGERS,
    ) -> None:
        KernelInputDeclaration.__init__(self, name, dtype)
        object.__setattr__(self, "z_staggers", frozenset(Stagger(s) for s in z_staggers))
        object.__setattr__(self, "y_staggers", frozenset(Stagger(s) for s in y_staggers))
        object.__setattr__(self, "x_staggers", frozenset(Stagger(s) for s in x_staggers))

    def validate_field(self, field: Field) -> None:
        """Validate that a field matches this declaration."""
        if field.output_dtype != self.dtype:
            raise TypeError(
                f"Kernel field data '{self.name}' has dtype '{self.dtype}', but field has dtype '{field.output_dtype}'."
            )
        if field.z_stagger not in self.z_staggers:
            raise ValueError(
                f"Valid z_staggers for kernel field data '{self.name}' are "
                f"{self.z_staggers}, "
                f"but field has z_stagger '{field.z_stagger}'."
            )
        if field.y_stagger not in self.y_staggers:
            raise ValueError(
                f"Valid y_staggers for kernel field data '{self.name}' are "
                f"{self.y_staggers}, "
                f"but field has y_stagger '{field.y_stagger}'."
            )
        if field.x_stagger not in self.x_staggers:
            raise ValueError(
                f"Valid x_staggers for kernel field data '{self.name}' are "
                f"{self.x_staggers}, "
                f"but field has x_stagger '{field.x_stagger}'."
            )

    @property
    def doc_string_part(self) -> str:
        return (
            f"'{self.name}' ({self.dtype})\n"
            f"    z_staggers={self.z_staggers}\n"
            f"    y_staggers={self.y_staggers}\n"
            f"    x_staggers={self.x_staggers}"
        )


class ParticleKernel:
    """A kernel to be execute on particles."""

    def __init__(
        self,
        fn: KernelFunction | Iterable[KernelFunction],
        particle_properties: Iterable[ParticlePropertyDeclaration] = (),
        scalars: Iterable[ScalarDeclaration] = (),
        field_data: Iterable[FieldDataDeclaration] = (),
    ):
        # store kernels as tuple of functions
        if callable(fn):
            funcs = (fn,)
        else:
            funcs = tuple(fn)

        if not all(callable(f) for f in funcs):
            raise TypeError("All kernel functions must be callable")
        self._funcs: tuple[KernelFunction, ...] = funcs

        # collect declarations
        particle_properties = tuple(particle_properties)
        scalars = tuple(scalars)
        field_data = tuple(field_data)

        # store declarations as dicts
        self._particle_properties = {p.name: p for p in particle_properties}
        self._scalars = {s.name: s for s in scalars}
        self._field_data = {f.name: f for f in field_data}

        # check for duplicate names
        if len(self._particle_properties) != len(particle_properties):
            raise ValueError("Duplicate particle property names in kernel declarations.")
        if len(self._scalars) != len(scalars):
            raise ValueError("Duplicate scalar names in kernel declarations.")
        if len(self._field_data) != len(field_data):
            raise ValueError("Duplicate field data names in kernel declarations.")

        # build docstring
        self.__doc__ = self._build_doc_string()

    @property
    def functions(self) -> tuple[KernelFunction, ...]:
        """The kernel functions."""
        return self._funcs

    @property
    def particle_properties(self) -> Mapping[str, ParticlePropertyDeclaration]:
        """The particle properties required by this kernel."""
        return types.MappingProxyType(self._particle_properties)

    @property
    def scalars(self) -> Mapping[str, ScalarDeclaration]:
        """The scalars required by this kernel."""
        return types.MappingProxyType(self._scalars)

    @property
    def field_data(self) -> Mapping[str, FieldDataDeclaration]:
        """The field data required by this kernel."""
        return types.MappingProxyType(self._field_data)

    @staticmethod
    def func_name(fn: KernelFunction) -> str:
        return getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))

    def __repr__(self) -> str:
        funcs = ", ".join(self.func_name(fn) for fn in self._funcs)

        return (
            f"{self.__class__.__name__}("
            f"funcs=[{funcs}], "
            f"particle_properties={list(self.particle_properties.values())}, "
            f"scalars={list(self.scalars.values())}, "
            f"field_data={list(self.field_data.values())}"
            f")"
        )

    def __str__(self) -> str:
        return "Particle Kernel: " + " → ".join(self.func_name(fn) for fn in self._funcs)

    def __call__(
        self,
        particle_properties: Mapping[str, npt.NDArray],
        scalars: Mapping[str, np.generic],
        field_data: Mapping[str, FieldData],
    ) -> None:
        """Execute the kernel."""
        for fn in self._funcs:
            fn(particle_properties, scalars, field_data)

    def _build_doc_string(self) -> str:
        doc_lines = ["Particle Kernel"]

        doc_lines.extend(_new_doc_section("Functions"))
        doc_lines.extend(self.func_name(fn) for fn in self._funcs)

        doc_lines.extend(_new_doc_section("Particle Properties"))
        doc_lines.extend(decl.doc_string_part for decl in self._particle_properties.values())

        doc_lines.extend(_new_doc_section("Scalars"))
        doc_lines.extend(decl.doc_string_part for decl in self._scalars.values())

        doc_lines.extend(_new_doc_section("Field Data"))
        doc_lines.extend(decl.doc_string_part for decl in self._field_data.values())

        return "\n".join(doc_lines)

    def bind(
        self,
        *,
        particle_properties: Mapping[str, str] | None = None,
        scalars: Mapping[str, str] | None = None,
        field_data: Mapping[str, str] | None = None,
    ) -> BoundKernel:
        """Create a BoundKernel for this kernel."""
        return BoundKernel(
            self,
            particle_property_bindings=particle_properties,
            scalar_bindings=scalars,
            field_data_bindings=field_data,
        )

    @classmethod
    def chain(cls, *Kernels: Self) -> Self:
        """Create a new ParticleKernel by merging kernels."""
        # concatenate functions
        funcs = sum((k._funcs for k in Kernels), ())
        # merge declarations
        particle_properties = _merge_declaration_dicts(*(k.particle_properties for k in Kernels))
        scalars = _merge_declaration_dicts(*(k.scalars for k in Kernels))
        field_data = _merge_declaration_dicts(*(k.field_data for k in Kernels))

        return cls(
            funcs,
            particle_properties.values(),
            scalars.values(),
            field_data.values(),
        )

    def chain_with(
        self,
        *others: Self,
    ) -> Self:
        """Chain this kernel with another kernel."""
        return self.__class__.chain(self, *others)


class BoundKernel:
    """An interface class binding kernel inputs to argument names."""

    def __init__(
        self,
        kernel: ParticleKernel,
        particle_property_bindings: Mapping[str, str] | None = None,
        scalar_bindings: Mapping[str, str] | None = None,
        field_data_bindings: Mapping[str, str] | None = None,
    ):
        self._kernel = kernel

        # defaults for optional arguments
        if particle_property_bindings is None:
            particle_property_bindings = {}
        if scalar_bindings is None:
            scalar_bindings = {}
        if field_data_bindings is None:
            field_data_bindings = {}

        # copy bindings to allow modification
        particle_property_bindings = dict(particle_property_bindings)
        scalar_bindings = dict(scalar_bindings)
        field_data_bindings = dict(field_data_bindings)

        # bind inputs defaulting to the declared names
        self._particle_property_bindings = {
            name: particle_property_bindings.pop(name, name) for name in kernel.particle_properties
        }
        self._scalar_bindings = {name: scalar_bindings.pop(name, name) for name in kernel.scalars}
        self._field_data_bindings = {name: field_data_bindings.pop(name, name) for name in kernel.field_data}

        # error if unused bindings
        if particle_property_bindings:
            raise ValueError(f"Unused particle property bindings: {particle_property_bindings}")
        if scalar_bindings:
            raise ValueError(f"Unused scalar bindings: {scalar_bindings}")
        if field_data_bindings:
            raise ValueError(f"Unused field data bindings: {field_data_bindings}")

        # build docstring
        self.__doc__ = self._build_doc_string()

    @property
    def kernel(self) -> ParticleKernel:
        """The underlying ParticleKernel."""
        return self._kernel

    @property
    def particle_property_bindings(self) -> Mapping[str, str]:
        """The particle property bindings."""
        return types.MappingProxyType(self._particle_property_bindings)

    @property
    def scalar_bindings(self) -> Mapping[str, str]:
        """The scalar bindings."""
        return types.MappingProxyType(self._scalar_bindings)

    @property
    def field_data_bindings(self) -> Mapping[str, str]:
        """The field data bindings."""
        return types.MappingProxyType(self._field_data_bindings)

    def rebind(
        self,
        *,
        particle_properties: Mapping[str, str] | None = None,
        scalars: Mapping[str, str] | None = None,
        field_data: Mapping[str, str] | None = None,
    ) -> Self:
        """Create a new BoundKernel with updated bindings."""
        new_particle_property_bindings = self._particle_property_bindings.copy()
        new_scalar_bindings = self._scalar_bindings.copy()
        new_field_data_bindings = self._field_data_bindings.copy()

        if particle_properties is not None:
            new_particle_property_bindings.update(particle_properties)
        if scalars is not None:
            new_scalar_bindings.update(scalars)
        if field_data is not None:
            new_field_data_bindings.update(field_data)

        return self.__class__(
            self._kernel,
            new_particle_property_bindings,
            new_scalar_bindings,
            new_field_data_bindings,
        )

    def _build_doc_string(self) -> str:
        kernel = self._kernel
        doc_lines = [f"Kernel Binding for {kernel}"]

        doc_lines.extend(_new_doc_section("Particle Property Bindings"))
        for name, binding in self._particle_property_bindings.items():
            doc_lines.append(f"'{binding}' → {kernel.particle_properties[name].doc_string_part}")

        doc_lines.extend(_new_doc_section("Scalar Bindings"))
        for name, binding in self._scalar_bindings.items():
            doc_lines.append(f"'{binding}' → {kernel.scalars[name].doc_string_part}")

        doc_lines.extend(_new_doc_section("Field Data Bindings"))
        for name, binding in self._field_data_bindings.items():
            doc_lines.append(f"'{binding}' → {kernel.field_data[name].doc_string_part}")

        return "\n".join(doc_lines)

    @classmethod
    def chain(cls, *Kernels: Self) -> Self:
        """Create a new BoundKernel by merging kernels."""
        # chain underlying kernels
        chained_kernel = ParticleKernel.chain(*(k._kernel for k in Kernels))
        # merge bindings
        particle_property_bindings = _merge_binding_dicts(*(k._particle_property_bindings for k in Kernels))
        scalar_bindings = _merge_binding_dicts(*(k._scalar_bindings for k in Kernels))
        field_data_bindings = _merge_binding_dicts(*(k._field_data_bindings for k in Kernels))

        return cls(
            chained_kernel,
            particle_property_bindings,
            scalar_bindings,
            field_data_bindings,
        )

    def chain_with(
        self,
        *others: Self,
    ) -> Self:
        """Chain this bound kernel with other bound kernels."""
        return self.__class__.chain(self, *others)


def get_required_particle_properties(*bound_kernels: BoundKernel) -> Mapping[str, ParticlePropertyDeclaration]:
    """Get the required particle properties from bound kernels.

    Args:
        bound_kernels: The bound kernels to get the required particle properties from.

    Returns:
        A mapping of required particle property names to their declarations.

    Raises:
        ValueError: If there are conflicting declarations for the same particle property name.
    """
    required: dict[str, ParticlePropertyDeclaration] = {}
    for kb in bound_kernels:
        for name in kb.particle_property_bindings:
            binding = kb.particle_property_bindings[name]
            particle_property = kb.kernel.particle_properties[name]
            if binding in required:
                if required[binding] != particle_property:
                    raise ValueError(
                        f"Conflicting declarations for particle property '{binding}': "
                        f"{required[binding]} vs {particle_property}"
                    )
            else:
                required[binding] = particle_property
    return types.MappingProxyType(required)


# helper functions


def _merge_declaration_dicts[KID: KernelInputDeclaration](*dicts: Mapping[str, KID]) -> dict[str, KID]:
    """Merge multiple declaration dicts, checking for conflicts."""
    merged: dict[str, KID] = {}
    for d in dicts:
        for key, decl in d.items():
            if key in merged:
                if merged[key] != decl:
                    raise ValueError(f"Conflicting declarations for '{key}': {merged[key]} vs {decl}")
            else:
                merged[key] = decl
    return merged


def _merge_binding_dicts(*dicts: Mapping[str, str]) -> dict[str, str]:
    """Merge multiple binding dicts, checking for conflicts."""
    merged: dict[str, str] = {}
    for d in dicts:
        for name, binding in d.items():
            if name in merged:
                if merged[name] != binding:
                    raise ValueError(f"Conflicting bindings for '{name}': {merged[name]} vs {binding}")
            else:
                merged[name] = binding
    return merged


def _new_doc_section(title: str) -> list[str]:
    return ["", title, "--------------"]
