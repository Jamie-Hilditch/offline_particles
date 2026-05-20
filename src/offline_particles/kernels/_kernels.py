"""Particle Kernels."""

from __future__ import annotations

import dataclasses
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

    def __init__(self, name: str, dtype_constraints: type[np.generic] | Iterable[type[np.generic]]) -> None:
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

    @property
    def _constraint_str(self) -> str:
        return " | ".join(str(dtype) for dtype in self.dtype_constraints)

    @property
    def _doc_string_part(self) -> str:
        return f"'{self.name}' ({self._constraint_str})"

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
    ) -> None:
        if callable(layout_validators):
            validators = (layout_validators,)
        else:
            validators = tuple(layout_validators)

        KernelInputDeclaration.__init__(self, name, dtype_constraints)
        object.__setattr__(self, "_layout_validators", validators)

    def validate_field(self, field: Field) -> None:
        """Validate that a field matches this declaration.

        Parameters
        ----------
        field : Field
            The field to validate.

        Raises
        ------
        TypeError
            If the field's dtype does not match the declaration's dtype or if the field does not satisfy the layout constraints.
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
    def _doc_string_part(self) -> str:
        return f"'{self.name}' ({self._constraint_str}) with {len(self._layout_validators)} layout validators"


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
        doc_lines.extend(decl._doc_string_part for decl in self._particle_properties.values())

        doc_lines.extend(_new_doc_section("Scalars"))
        doc_lines.extend(decl._doc_string_part for decl in self._scalars.values())

        doc_lines.extend(_new_doc_section("Field Data"))
        doc_lines.extend(decl._doc_string_part for decl in self._field_data.values())

        return "\n".join(doc_lines)

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

        # build docstring
        self.__doc__ = self._build_doc_string()

    @property
    def kernel(self) -> ParticleKernel:
        """The underlying ParticleKernel."""
        return self._kernel

    @property
    def particle_property_bindings(self) -> Mapping[str, str]:
        """The particle property bindings.

        A mapping from declared names to bound names.
        """
        return types.MappingProxyType(self._particle_property_bindings)

    @property
    def scalar_bindings(self) -> Mapping[str, str]:
        """The scalar bindings.

        A mapping from declared names to bound names.
        """
        return types.MappingProxyType(self._scalar_bindings)

    @property
    def field_data_bindings(self) -> Mapping[str, str]:
        """The field data bindings.

        A mapping from declared names to bound names.
        """
        return types.MappingProxyType(self._field_data_bindings)

    @property
    def particle_property_declarations(self) -> dict[str, ParticlePropertyDeclaration]:
        """The particle property declarations.

        A mapping from bound names to declarations.
        """
        return {
            bound_name: self.kernel.particle_properties[declared_name]
            for declared_name, bound_name in self.particle_property_bindings.items()
        }

    @property
    def scalar_declarations(self) -> dict[str, ScalarDeclaration]:
        """The scalar declarations.

        A mapping from bound names to declarations.
        """
        return {
            bound_name: self.kernel.scalars[declared_name] for declared_name, bound_name in self.scalar_bindings.items()
        }

    @property
    def field_data_declarations(self) -> dict[str, FieldDataDeclaration]:
        """The field data declarations.

        A mapping from bound names to declarations.
        """
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

    def _build_doc_string(self) -> str:
        kernel = self._kernel
        doc_lines = [f"Kernel Binding for {kernel}"]

        doc_lines.extend(_new_doc_section("Particle Property Bindings"))
        for bound_name, declaration in self.particle_property_declarations.items():
            doc_lines.append(f"'{bound_name}' → {declaration._doc_string_part}")

        doc_lines.extend(_new_doc_section("Scalar Bindings"))
        for bound_name, declaration in self.scalar_declarations.items():
            doc_lines.append(f"'{bound_name}' → {declaration._doc_string_part}")

        doc_lines.extend(_new_doc_section("Field Data Bindings"))
        for bound_name, declaration in self.field_data_declarations.items():
            doc_lines.append(f"'{bound_name}' → {declaration._doc_string_part}")

        return "\n".join(doc_lines)

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


def _new_doc_section(title: str) -> list[str]:
    return ["", title, "--------------"]
