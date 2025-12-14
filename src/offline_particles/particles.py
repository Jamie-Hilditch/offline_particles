"""The particles."""

from typing import Any

import numpy as np
import numpy.typing as npt


class Particles:
    __slots__ = ("_length", "_arrays", "_frozen")

    def __init__(self, nparticles: int, **fields: npt.DTypeLike) -> None:
        """Initialize the Particles object.

        Args:
            nparticles: The number of particles.
            **fields: The particle fields and their dtypes.
        """
        self._length = nparticles
        self._arrays = {
            field: np.zeros((nparticles,), dtype=dtype)
            for field, dtype in fields.items()
        }

        # make the object immutable
        self._frozen = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"'Particles' object is immutable; cannot set attribute '{name}'"
            )
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> npt.NDArray:
        if name in self._arrays:
            return self._arrays[name]
        raise AttributeError(f"'Particles' object has no attribute '{name}'")

    def __getitem__(self, name: str) -> npt.NDArray:
        return self._arrays[name]

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}:{arr.dtype}" for name, arr in self._arrays.items())
        return (
            f"{self.__class__.__name__}("
            f"nparticles={self._length}, "
            f"fields={{ {fields} }}"
            f")"
        )

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
            f"(nparticles={self._length}, "
            f"fields=[{public_fields}]"
            f"{hidden_str}"
            f")"
        )
