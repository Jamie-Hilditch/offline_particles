"""Particle Kernels."""

from __future__ import annotations

import dataclasses
import functools
import types
from collections.abc import Callable, Iterable, Mapping
from typing import Self

import numpy as np
import numpy.typing as npt

from ..fields import Field, FieldData
from ..spatial_arrays import ArrayLayout

# these type aliases are manually documented in the module docstring for better formatting in the docs,
# if they are updated here, also update the docstring at the top of the __init__.py file
type ParticlePropertiesType = Mapping[str, npt.NDArray]
type ScalarsType = Mapping[str, np.generic]
type FieldDataType = Mapping[str, FieldData]
type KernelFunction = Callable[[ParticlePropertiesType, ScalarsType, FieldDataType], None]

type LayoutValidator = Callable[[ArrayLayout], None]


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class KernelInputDeclaration:
    """Declaration of a kernel input."""

    name: str
    dtype_constraints: tuple[type[np.generic], ...]
    _description: str = dataclasses.field(compare=False)

    def __init__(
        self, name: str, dtype_constraints: type[np.generic] | Iterable[type[np.generic]], description: str = ""
    ) -> None:
        match dtype_constraints:
            case type() as dtype:
                dtypes = (dtype,)
            case Iterable() as dtypes:
                dtypes = tuple(dtypes)
            case _:
                raise TypeError("dtype_constraints must be a type or an iterable of types")
        for dtype in dtypes:
            if not isinstance(dtype, type) or not issubclass(dtype, np.generic):
                raise TypeError(f"dtype_constraints must be a subtype of np.generic, got {dtype}")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype_constraints", dtypes)
        object.__setattr__(self, "_description", description)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, dtype_constraints=[{self._constraint_str.replace(' | ', ', ')}], description={self._description!r})"

    def __str__(self) -> str:
        return self.summary

    @property
    def _constraint_str(self) -> str:
        return " | ".join(getattr(dtype, "__name__", str(dtype)) for dtype in self.dtype_constraints)

    @property
    def summary(self) -> str:
        return f"{self.name} : {self._constraint_str}"

    @property
    def description(self) -> str:
        if not self._description:
            return self.summary
        return self.summary + "\n\t" + self._description

    def validate_dtype(self, dtype: type[np.generic] | np.dtype) -> None:
        """Validate that the dtype satisfies the declared constraints.

        Parameters
        ----------
        dtype : type[np.generic] | np.dtype
            The dtype to validate.

        Raises
        ------
        TypeError
            If the dtype does not satisfy the declaration's dtype constraints.
        """
        if not any(np.issubdtype(dtype, constraint) for constraint in self.dtype_constraints):
            raise TypeError(
                f"Kernel input '{self.name}' has dtype constraints `{self._constraint_str}`, but the provided dtype '{dtype}' does not satisfy these constraints."
            )


class ParticlePropertyDeclaration(KernelInputDeclaration):
    """Declaration of a particle property required by a kernel."""


class ScalarDeclaration(KernelInputDeclaration):
    """Declaration of a scalar required by a kernel."""


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class FieldDataDeclaration(KernelInputDeclaration):
    """Declaration of field data required by a kernel."""

    _layout_validators: tuple[LayoutValidator, ...]

    def __init__(
        self,
        name: str,
        dtype_constraints: type[np.generic] | Iterable[type[np.generic]],
        layout_validators: LayoutValidator | Iterable[LayoutValidator] = (),
        description: str = "",
    ) -> None:
        if callable(layout_validators):
            validators = (layout_validators,)
        else:
            validators = tuple(layout_validators)

        KernelInputDeclaration.__init__(self, name, dtype_constraints, description=description)
        object.__setattr__(self, "_layout_validators", validators)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"dtype_constraints=[{self._constraint_str.replace(' | ', ', ')}], "
            f"layout validators={self._layout_validators!r}, "
            f"description={self._description!r})"
        )

    def validate_field(self, field: Field) -> None:
        """Validate that a field matches this declaration.

        Parameters
        ----------
        field : Field
            The field to validate.

        Raises
        ------
        TypeError
            If the field's dtype does not match the declaration's dtype.
        ValueError
            If the field does not satisfy the layout constraints.
        """
        try:
            self.validate_dtype(field.output_dtype)
        except TypeError as e:
            raise TypeError(
                f"Input Field does not satisfy the dtype constraints for kernel field data '{self.name}'"
            ) from e

        for validator in self._layout_validators:
            try:
                validator(field.layout)
            except Exception as e:
                raise ValueError(
                    f"Input Field does not satisfy the layout constraints for kernel field data '{self.name}'"
                ) from e

    @property
    def summary(self) -> str:
        return f"{self.name} : {self._constraint_str} with {len(self._layout_validators)} layout validators"


class ParticleKernel:
    """A kernel to be execute on particles.

    Parameters
    ----------
    fn : KernelFunction or Iterable[KernelFunction]
        The kernel function(s) to execute. If multiple functions are provided, they will be executed in the order given.
    particle_properties : Iterable[ParticlePropertyDeclaration], optional
        Declarations of the particle properties required by this kernel. Default is an empty iterable.
    scalars : Iterable[ScalarDeclaration], optional
        Declarations of the scalars required by this kernel. Default is an empty iterable.
    field_data : Iterable[FieldDataDeclaration], optional
        Declarations of the field data required by this kernel. Default is an empty iterable.
    name : str, optional
        An optional name for the kernel. This is used in the summary and description of the kernel.
        Default is None, which results in a generic summary and description without a name.
    """

    def __init__(
        self,
        fn: KernelFunction | Iterable[KernelFunction],
        particle_properties: Iterable[ParticlePropertyDeclaration] = (),
        scalars: Iterable[ScalarDeclaration] = (),
        field_data: Iterable[FieldDataDeclaration] = (),
        name: str | None = None,
    ):
        self._name = name

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

    @property
    def name(self) -> str | None:
        """The name of the kernel, or None if not specified."""
        return self._name

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

    @staticmethod
    def func_doc(fn: KernelFunction) -> str:
        return getattr(fn, "__doc__", "")

    @staticmethod
    def func_summary(fn: KernelFunction) -> str:
        """Extract a summary from a function's docstring.

        Returns
        -------
        str
            The first line of the function's docstring as a summary, or return an empty string if no docstring.
        """
        doc = ParticleKernel.func_doc(fn)
        if not doc:
            return ""
        return doc.strip().splitlines()[0]

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
        return self.summary

    def __call__(
        self,
        particle_properties: Mapping[str, npt.NDArray],
        scalars: Mapping[str, np.generic],
        field_data: Mapping[str, FieldData],
    ) -> None:
        """Execute the kernel."""
        for fn in self._funcs:
            fn(particle_properties, scalars, field_data)

    @property
    def summary(self) -> str:
        name_part = f"{self.name} : " if self.name else "Particle Kernel : "
        return name_part + " → ".join(self.func_name(fn) for fn in self._funcs)

    @property
    def description(self) -> str:
        """A detailed, multi-line description of the kernel's functions and required inputs."""
        description_lines = [f"Particle Kernel: {self.name}"] if self.name else ["Particle Kernel"]

        # functions
        description_lines.extend(_new_description_section("Functions"))
        for fn in self._funcs:
            description_lines.append(self.func_name(fn))
            func_summary = self.func_summary(fn)
            if func_summary:
                description_lines.append("\t" + func_summary)

        # particle properties
        description_lines.extend(_new_description_section("Particle Properties"))
        description_lines.extend(decl.description for decl in self._particle_properties.values())

        # scalars
        description_lines.extend(_new_description_section("Scalars"))
        description_lines.extend(decl.description for decl in self._scalars.values())

        # field data
        description_lines.extend(_new_description_section("Field Data"))
        description_lines.extend(decl.description for decl in self._field_data.values())

        return "\n".join(description_lines)

    def bind(
        self,
        *,
        particle_properties: Mapping[str, str] | None = None,
        scalars: Mapping[str, str] | None = None,
        field_data: Mapping[str, str] | None = None,
    ) -> BoundKernel:
        """Create a BoundKernel for this kernel.

        Parameters
        ----------
        particle_properties : Mapping[str, str], optional
            A mapping of declared particle property names to actual names. If not provided, the declared names are used.
        scalars : Mapping[str, str], optional
            A mapping of declared scalar names to argument names. If not provided, the declared names are used.
        field_data : Mapping[str, str], optional
            A mapping of declared field data names to argument names. If not provided, the declared names are used.

        Returns
        -------
        BoundKernel
            A BoundKernel with the specified bindings.
        """
        return BoundKernel(
            self,
            particle_property_bindings=particle_properties,
            scalar_bindings=scalars,
            field_data_bindings=field_data,
        )

    @classmethod
    def chain(cls, *Kernels: Self) -> Self:
        """Create a new ParticleKernel by merging kernels.

        Parameters
        ----------
        *Kernels : ParticleKernel
            The kernels to merge.

        Returns
        -------
        ParticleKernel
            A new ParticleKernel that combines the functions and declarations of the input kernels.
        """
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
        """Chain this kernel with another kernel.

        Parameters
        ----------
        *others : ParticleKernel
            Other kernels to chain with this kernel.

        Returns
        -------
        ParticleKernel
            A new ParticleKernel that combines the functions and declarations of this kernel and the others.
        """
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
        """Create a bound kernel.

        Parameters
        ----------
        kernel : ParticleKernel
            The kernel to bind.
        particle_property_bindings : Mapping[str, str], optional
            A mapping of declared particle property names to bound names. If not provided, the declared names are used as bound names.
        scalar_bindings : Mapping[str, str], optional
            A mapping of declared scalar names to bound names. If not provided, the declared names are used as bound names.
        field_data_bindings : Mapping[str, str], optional
            A mapping of declared field data names to bound names. If not provided, the declared names are used as bound names.

        Raises
        ------
        ValueError
            If there are unused bindings in any of the provided binding mappings.
        """
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
            declared_name: particle_property_bindings.pop(declared_name, declared_name)
            for declared_name in kernel.particle_properties
        }
        self._scalar_bindings = {
            declared_name: scalar_bindings.pop(declared_name, declared_name) for declared_name in kernel.scalars
        }
        self._field_data_bindings = {
            declared_name: field_data_bindings.pop(declared_name, declared_name) for declared_name in kernel.field_data
        }

        # error if unused bindings
        if particle_property_bindings:
            raise ValueError(f"Unused particle property bindings: {particle_property_bindings}")
        if scalar_bindings:
            raise ValueError(f"Unused scalar bindings: {scalar_bindings}")
        if field_data_bindings:
            raise ValueError(f"Unused field data bindings: {field_data_bindings}")

    @property
    def kernel(self) -> ParticleKernel:
        """The underlying ParticleKernel."""
        return self._kernel

    @property
    def particle_property_bindings(self) -> Mapping[str, str]:
        """A mapping from declared particle property names to bound names."""
        return types.MappingProxyType(self._particle_property_bindings)

    @property
    def scalar_bindings(self) -> Mapping[str, str]:
        """A mapping from declared scalar names to bound names."""
        return types.MappingProxyType(self._scalar_bindings)

    @property
    def field_data_bindings(self) -> Mapping[str, str]:
        """A mapping from declared field data names to bound names."""
        return types.MappingProxyType(self._field_data_bindings)

    @property
    def particle_property_declarations(self) -> dict[str, ParticlePropertyDeclaration]:
        """A mapping from bound particle property names to their declarations."""
        return {
            bound_name: self.kernel.particle_properties[declared_name]
            for declared_name, bound_name in self.particle_property_bindings.items()
        }

    @property
    def scalar_declarations(self) -> dict[str, ScalarDeclaration]:
        """A mapping from bound scalar names to their declarations."""
        return {
            bound_name: self.kernel.scalars[declared_name] for declared_name, bound_name in self.scalar_bindings.items()
        }

    @property
    def field_data_declarations(self) -> dict[str, FieldDataDeclaration]:
        """A mapping from bound field data names to their declarations."""
        return {
            bound_name: self.kernel.field_data[declared_name]
            for declared_name, bound_name in self.field_data_bindings.items()
        }

    def validate_particles(self, particles: Mapping[str, npt.NDArray]) -> None:
        """Validate that the particles have required particle properties of the correct data types.

        Parameters
        ----------
        particles : Mapping[str, npt.NDArray]
            A mapping from particle property names to their corresponding arrays.

        Raises
        ------
        KeyError
            If the particles do not have a required particle property.
        TypeError
            If a particle property has a dtype that does not satisfy the declaration's dtype constraints.
            From :meth:`~ParticlePropertyDeclaration.validate_dtype` for each particle property.
        """
        for declared_name, bound_name in self._particle_property_bindings.items():
            if bound_name not in particles:
                raise KeyError(f"Particle property '{bound_name}' is required but not found in the particles.")
            # get declaration and validate dtype
            particle_property_declaration = self._kernel.particle_properties[declared_name]
            particle_property_declaration.validate_dtype(particles[bound_name].dtype)

    def rebind(
        self,
        *,
        particle_properties: Mapping[str, str] | None = None,
        scalars: Mapping[str, str] | None = None,
        field_data: Mapping[str, str] | None = None,
    ) -> Self:
        """Create a new BoundKernel with updated bindings.

        Parameters
        ----------
        particle_properties : Mapping[str, str], optional
            A mapping of declared particle property names to new bound names. If not provided, the current bindings are used.
        scalars : Mapping[str, str], optional
            A mapping of declared scalar names to new bound names. If not provided, the current bindings are used.
        field_data : Mapping[str, str], optional
            A mapping of declared field data names to new bound names. If not provided, the current bindings are used.

        Returns
        -------
        BoundKernel
            A new BoundKernel with the updated bindings.
        """
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

    @property
    def summary(self) -> str:
        """A one-line summary of the bound kernel."""
        return f"Binding for {self.kernel.summary}"

    @property
    def description(self) -> str:
        """A detailed, multi-line description of the bound kernel, including the underlying kernel's description and bindings."""
        description_lines = [f"Bound Kernel: {self.kernel.name}"] if self.kernel.name else ["Bound Kernel"]

        # particle property bindings
        description_lines.extend(_new_description_section("Particle Property Bindings"))
        for declared_name, bound_name in self.particle_property_bindings.items():
            description_lines.append(f"{declared_name} ← {bound_name}")

        # scalar bindings
        description_lines.extend(_new_description_section("Scalar Bindings"))
        for declared_name, bound_name in self.scalar_bindings.items():
            description_lines.append(f"{declared_name} ← {bound_name}")

        # field data bindings
        description_lines.extend(_new_description_section("Field Data Bindings"))
        for declared_name, bound_name in self.field_data_bindings.items():
            description_lines.append(f"{declared_name} ← {bound_name}")

        # kernel description
        description_lines.extend(_new_description_section("Particle Kernel Description"))
        description_lines.append(self.kernel.description)

        return "\n".join(description_lines)

    @classmethod
    def chain(cls, *Kernels: Self) -> Self:
        """Create a new BoundKernel by merging kernels.

        Parameters
        ----------
        *Kernels : BoundKernel
            The bound kernels to merge.

        Returns
        -------
        BoundKernel
            A new BoundKernel that combines the underlying kernels and their bindings.
        """
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
        """Chain this bound kernel with other bound kernels.

        Parameters
        ----------
        *others : BoundKernel
            Other bound kernels to chain with this bound kernel.

        Returns
        -------
        BoundKernel
            A new BoundKernel that combines the underlying kernels and their bindings.
        """
        return self.__class__.chain(self, *others)


# decorators


def kernel_function(
    particle_property_keys: Iterable[str] = (),
    scalar_keys: Iterable[str] = (),
    field_data_keys: Iterable[str] = (),
) -> Callable[[Callable], KernelFunction]:
    """Convert a kernel function implementation into a kernel function.

    This decorator returns a ``KernelFunction`` that unpacks the kernel input dictionaries into individual arrays and scalars
    and then passes them to the decorated function. This particularly useful for implementing kernel functions with numba.

    Parameters
    ----------
    particle_property_keys : Iterable[str], optional
        The names of the particle properties required by this kernel function.
    scalar_keys : Iterable[str], optional
        The names of the scalars required by this kernel function.
    field_data_keys : Iterable[str], optional
        The names of the field data required by this kernel function.

    Returns
    -------
    Callable[[Callable], KernelFunction]
        A decorator that converts a kernel function implementation into a kernel function.

    Notes
    -----
    The decorated function should have a signature that accepts the unpacked particle properties, scalars, and field data
    in the order specified by the keys. The field data is itself unpacked into its components (array and offsets) and passed as separate arguments.
    For example, if a field data key corresponds to a FieldData with 3D offsets, the decorated function will receive 4 arguments
    for that field data: the array and the 3 offsets.
    """

    def decorator(fn: Callable) -> KernelFunction:
        @functools.wraps(fn)
        def _kernel_function(
            particle_properties: ParticlePropertiesType,
            scalars: ScalarsType,
            field_data: FieldDataType,
        ) -> None:
            particle_property_args = (particle_properties[name] for name in particle_property_keys)
            scalar_args = (scalars[name] for name in scalar_keys)
            field_data_args = (arg for name in field_data_keys for arg in field_data[name].unpack())
            return fn(*particle_property_args, *scalar_args, *field_data_args)

        return _kernel_function

    return decorator


# helper functions


def _merge_declaration_dicts[KID: KernelInputDeclaration](*dicts: Mapping[str, KID]) -> dict[str, KID]:
    """Merge multiple declaration dicts, checking for conflicts.

    Parameters
    ----------
    *dicts : Mapping[str, KID]
        The declaration dicts to merge.

    Returns
    -------
    dict[str, KID]
        A merged declaration dict.

    Raises
    ------
    ValueError
        If there are conflicting declarations for the same name.
    """
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
    """Merge multiple binding dicts, checking for conflicts.

    Parameters
    ----------
    *dicts : Mapping[str, str]
        The binding dicts to merge.

    Returns
    -------
    dict[str, str]
        A merged binding dict.

    Raises
    ------
    ValueError
        If there are conflicting bindings for the same name.
    """
    merged: dict[str, str] = {}
    for d in dicts:
        for name, binding in d.items():
            if name in merged:
                if merged[name] != binding:
                    raise ValueError(f"Conflicting bindings for '{name}': {merged[name]} vs {binding}")
            else:
                merged[name] = binding
    return merged


def _new_description_section(title: str) -> list[str]:
    return ["", title, "-" * len(title)]
