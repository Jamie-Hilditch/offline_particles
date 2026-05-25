"""The particles."""

import types
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from .kernels import BoundKernel, ParticlePropertyDeclaration

_REQUIRED_PARTICLE_PROPERTY_FIELDS = {
    "status": np.dtype(np.uint8),
    "zidx": np.dtype(np.float64),
    "yidx": np.dtype(np.float64),
    "xidx": np.dtype(np.float64),
}


class _FrozenArrayMapping:
    """A mapping-like object that holds equi-shaped arrays and prevents modification."""

    __slots__ = ("_arrays", "_dtypes", "_shape")

    def __init__(self, arrays: Mapping[str, npt.NDArray]) -> None:
        """Initialize the mapping with given arrays.

        Parameters
        ----------
        arrays : Mapping[str, npt.NDArray]
            The arrays to store in the mapping.

        Raises
        ------
        ValueError
            If the arrays do not all have the same shape.
        """
        shapes = {arr.shape for arr in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("All arrays must have the same shape. Got shapes: " + ", ".join(str(s) for s in shapes))
        object.__setattr__(self, "_shape", shapes.pop())
        object.__setattr__(self, "_dtypes", types.MappingProxyType({name: arr.dtype for name, arr in arrays.items()}))
        object.__setattr__(self, "_arrays", types.MappingProxyType(arrays))

    def __setattr__(self, name, value):
        raise AttributeError(f"{self.__class__.__name__} is immutable")

    def __getattr__(self, name: str) -> npt.NDArray:
        try:
            arrays = object.__getattribute__(self, "_arrays")
            return arrays[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, name: str) -> npt.NDArray:
        return object.__getattribute__(self, "_arrays")[name]

    def __contains__(self, name: str) -> bool:
        return name in object.__getattribute__(self, "_arrays")

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the arrays in the mapping."""
        return object.__getattribute__(self, "_shape")

    @property
    def arrays(self) -> Mapping[str, npt.NDArray]:
        """The arrays in the mapping."""
        return object.__getattribute__(self, "_arrays")

    @property
    def dtypes(self) -> Mapping[str, np.dtype]:
        """The dtypes of the arrays in the mapping."""
        return object.__getattribute__(self, "_dtypes")

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}:{dtype}" for name, dtype in self.dtypes.items())
        return f"{self.__class__.__name__}(shape={self.shape}, fields={{ {fields} }})"

    def __str__(self) -> str:
        public = [name for name in self._arrays if not name.startswith("_")]
        hidden_count = sum(1 for name in self._arrays if name.startswith("_"))

        public_fields = ", ".join(public)

        if hidden_count > 1:
            hidden_str = f", +{hidden_count} hidden fields"
        elif hidden_count == 1:
            hidden_str = ", +1 hidden field"
        else:
            hidden_str = ""

        return f"{self.__class__.__name__} (shape={self.shape}, fields=[{public_fields}]{hidden_str})"


class Particles(_FrozenArrayMapping):
    __slots__ = ("_length",)

    def __init__(self, nparticles: int, bound_property_dtypes: Mapping[str, np.dtype]) -> None:
        """Initialize the Particles object.

        Parameters
        ----------
        nparticles : int
            The number of particles.
        bound_property_dtypes : Mapping[str, np.dtype]
            Dtypes for additional particle properties to create. The required
            particle properties ``status``, ``zidx``, ``yidx``, and ``xidx``
            are always created with their required dtypes, even if they are
            not included in this mapping. If this mapping includes any of
            those required property names, the provided dtypes are only used
            for validation and must exactly match the required dtypes.

        Raises
        ------
        ValueError
            If a passed dtype is incompatible with the required dtypes.
        """
        object.__setattr__(self, "_length", nparticles)

        # sort the additional bindings alphabetically, but with private fields (those starting with "_") coming after public ones
        bindings = sorted(bound_property_dtypes.keys(), key=lambda binding: (binding.startswith("_"), binding))
        bindings_dtypes = {binding: np.dtype(bound_property_dtypes[binding]) for binding in bindings}

        # remove any required fields from the additional bindings, since we already created arrays for those and checked their dtypes
        for required_binding, dtype in _REQUIRED_PARTICLE_PROPERTY_FIELDS.items():
            provided_dtype = bindings_dtypes.pop(required_binding, None)
            if provided_dtype is None:
                continue
            if provided_dtype != dtype:
                raise ValueError(
                    f"Invalid dtype for required particle property '{required_binding}'. Provided dtype {provided_dtype} is not equal to required dtype {dtype}."
                )

        # build up arrays starting with the required fields
        arrays = {
            binding: np.zeros((nparticles,), dtype=dtype)
            for binding, dtype in _REQUIRED_PARTICLE_PROPERTY_FIELDS.items()
        }

        # now add the additional fields, checking that they are compatible with any arrays we already created for the required fields
        for binding, dtype in bindings_dtypes.items():
            arrays[binding] = np.zeros((nparticles,), dtype=dtype)

        super().__init__(arrays=arrays)

    def __len__(self) -> int:
        return object.__getattribute__(self, "_length")

    @classmethod
    def build_from_kernels(
        cls, nparticles: int, specified_dtypes: Mapping[str, npt.DTypeLike], kernels: list[BoundKernel]
    ) -> "Particles":
        """Initialize a Particles object from a list of particle property declarations.

        Parameters
        ----------
        nparticles : int
            The number of particles.
        specified_dtypes : Mapping[str, npt.DTypeLike]
            The specified dtypes for the particle properties.
        kernels : list[BoundKernel]
            The bound kernels to use for initializing the particles.

        Returns
        -------
        Particles
            A Particles object satisfying the constraints specified by the kernels.
        """
        # first collect all the names of the properties we need to initialize and the declarations that apply to them
        declarations_by_binding: dict[str, list[ParticlePropertyDeclaration]] = {}
        for kernel in kernels:
            for name, declaration in kernel.particle_property_declarations.items():
                decls = declarations_by_binding.setdefault(name, [])
                decls.append(declaration)

        # now, for each property, find a valid dtype that satisfies all of its declarations
        bound_property_dtypes = {}
        for name, declarations in declarations_by_binding.items():
            if name in specified_dtypes:
                # if the user specified a dtype for this property, we use it
                dtype = np.dtype(specified_dtypes[name])
                for declaration in declarations:
                    declaration.validate_dtype(dtype)
                bound_property_dtypes[name] = np.dtype(specified_dtypes[name])
            else:
                # otherwise, we find a valid dtype that satisfies all declarations for this property
                bound_property_dtypes[name] = _find_valid_particle_property_dtype(declarations)

        # construct the Particles object with the determined dtypes
        return cls(nparticles, bound_property_dtypes=bound_property_dtypes)


class ParticlesView(_FrozenArrayMapping):
    """A read-only view of particle arrays."""

    __slots__ = ("_length",)

    def __init__(self, parent: Particles) -> None:
        """Initialize the ParticlesView.

        Parameters
        ----------
        parent : Particles
            The parent :class:`Particles` object.
        """
        arrays = {name: self.readonly_view(array) for name, array in parent.arrays.items()}
        object.__setattr__(self, "_length", len(parent))
        super().__init__(arrays=arrays)

    @staticmethod
    def readonly_view(array: npt.NDArray) -> npt.NDArray:
        """Create a read-only view of the given array.

        Parameters
        ----------
        array : npt.NDArray
            The input array.

        Returns
        -------
        npt.NDArray
            A read-only view of the input array.
        """
        view = array.view()
        view.setflags(write=False)
        return view

    def __len__(self) -> int:
        return object.__getattribute__(self, "_length")


_CONCRETE_NUMPY_TYPES_BY_PREFERENCE = (
    # defaults that cover most use cases
    np.float64,
    np.float32,
    np.complex128,
    np.int32,
    np.uint8,
    np.dtype("datetime64[ns]"),
    np.dtype("timedelta64[ns]"),
    # rest of the float types
    np.float16,
    np.longdouble,
    # rest of the complex types
    np.complex64,
    np.clongdouble,
    # rest of the int types
    np.int8,
    np.int16,
    np.int64,
    # rest of the uint types
    np.uint16,
    np.uint32,
    np.uint64,
)


def _find_valid_particle_property_dtype(declarations: list[ParticlePropertyDeclaration]) -> np.dtype:
    """Find a valid dtype for a particle property given a set of declarations.

    Parameters
    ----------
    declarations : Iterable[ParticlePropertyDeclaration]
        The declarations to find a valid dtype for.

    Returns
    -------
    np.dtype
        A valid dtype that satisfies all declarations.

    Raises
    ------
    ValueError
        If no valid dtype can be found that satisfies all declarations.
    """
    for dtype in _CONCRETE_NUMPY_TYPES_BY_PREFERENCE:
        try:
            for declaration in declarations:
                declaration.validate_dtype(dtype)
            return np.dtype(dtype)
        except TypeError:
            continue

    constraint_strings = [d._constraint_str for d in declarations]
    constraint_message = "[ " + " ], [ ".join(constraint_strings) + " ]"
    raise ValueError(f"No valid dtype found for particle property with constraints {constraint_message}.")
