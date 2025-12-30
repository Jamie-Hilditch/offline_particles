"""The particles."""

import types
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt


class _FrozenArrayMapping:
    """A mapping-like object that holds equi-shaped arrays and prevents modification."""

    __slots__ = ("_shape", "_dtypes", "_arrays", "_frozen")

    def __init__(self, **arrays: npt.NDArray) -> None:
        """Initialize the mapping with given arrays.

        Args:
            **arrays: The arrays to store in the mapping.
        """
        shapes = {arr.shape for arr in arrays.values()}
        if len(shapes) != 1:
            raise ValueError(
                "All arrays must have the same shape. Got shapes: "
                + ", ".join(str(s) for s in shapes)
            )
        self._shape = shapes.pop()
        self._dtypes = types.MappingProxyType(
            {name: arr.dtype for name, arr in arrays.items()}
        )
        self._arrays = types.MappingProxyType(arrays)
        self._frozen = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"'{self.__class__.__name__}' object is immutable; cannot set attribute '{name}'"
            )
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> npt.NDArray:
        try:
            return self._arrays[name]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __getitem__(self, name: str) -> npt.NDArray:
        return self._arrays[name]

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the arrays in the mapping."""
        return self._shape

    @property
    def dtypes(self) -> Mapping[str, np.dtype]:
        """The dtypes of the arrays in the mapping."""
        return self._dtypes

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

        return (
            f"{self.__class__.__name__} "
            f"(shape={self.shape}, "
            f"fields=[{public_fields}]"
            f"{hidden_str}"
            f")"
        )


class Particles(_FrozenArrayMapping):
    __slots__ = ("_length",)

    def __init__(self, nparticles: int, **fields: npt.DTypeLike) -> None:
        """Initialize the Particles object.

        Args:
            nparticles: The number of particles.
            **fields: The particle fields and their dtypes.
        """
        self._length = nparticles
        arrays = {
            field: np.zeros((nparticles,), dtype=dtype)
            for field, dtype in fields.items()
        }

        super().__init__(**arrays)

    def __len__(self) -> int:
        return self._length


class ParticlesView(_FrozenArrayMapping):
    """A read-only view of particle arrays."""

    __slots__ = ("_length",)

    def __init__(self, parent: Particles) -> None:
        """Initialize the ParticlesView.

        Args:
            parent: The parent Particles object.
        """
        arrays = {
            name: self.readonly_view(parent[name]) for name in parent.dtypes.keys()
        }
        self._length = len(parent)
        super().__init__(**arrays)

    @staticmethod
    def readonly_view(array: npt.NDArray) -> npt.NDArray:
        """Create a read-only view of the given array.

        Args:
            array: The input array.
        """
        view = array.view()
        view.setflags(write=False)
        return view

    def __len__(self) -> int:
        return self._length
