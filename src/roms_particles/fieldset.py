"""Submodule for Fieldset, a collection of fields from a simulation."""

import itertools

import numpy.typing as npt

from numbers import Number
from typing import Iterable

from .fields import Field


class Fieldset:
    def __init__(
        self,
        t_size: int,
        z_size: int,
        y_size: int,
        x_size: int,
        time: npt.NDArray,
        fields: dict[str, Field],
        constants: dict[str, Number],
    ) -> None:
        # sizes of centered dimensions
        self._t_size = t_size
        self._z_size = z_size
        self._y_size = y_size
        self._x_size = x_size

        # check time array is correct length
        if len(time) != t_size:
            raise ValueError(f"time has length {len(time)} but t_size = {t_size}")

        self._time = time
        self._fields: dict[str, Field] = {}
        self._constants: dict[str, Number] = {}

        # add fields and constants
        for name, field in fields.items():
            self.add_field(name, field)

        for name, value in constants.items():
            self.add_constant(name, value)

    @property
    def t_size(self) -> int:
        """Size of the time dimension."""
        return self._t_size

    @property
    def z_size(self) -> int:
        """Size of the centered z dimension."""
        return self._z_size

    @property
    def y_size(self) -> int:
        """Size of the centered y dimension."""
        return self._y_size

    @property
    def x_size(self) -> int:
        """Size of the centered x dimension."""
        return self._x_size

    @property
    def simulation_shape(self) -> tuple[int, int, int, int]:
        """4D shape of the simulation assuming centered grids."""
        return (self._t_size, self._z_size, self._y_size, self._x_size)

    @property
    def time(self) -> npt.NDArray:
        """Time array of the fieldset."""
        return self._time

    @property
    def fields(self) -> dict[str, Field]:
        """Dictionary of fields in the fieldset."""
        return self._fields

    @property
    def constants(self) -> dict[str, Number]:
        """Dictionary of constants in the fieldset."""
        return self._constants

    def add_field(self, name: str, field: Field) -> None:
        """Add a field to the fieldset.
        Parameters:
            name: name of the field
            field: Field object
        """
        if name in self._fields or name in self._constants:
            raise KeyError(
                f"'{name}' already exists in Fieldset. First remove it before adding a new one."
            )
        try:
            field.validate_shape(self.simulation_shape)
        except ValueError as e:
            raise ValueError(f"Error validating shape of Field '{name}'.") from e
        self._fields[name] = field

    def remove_field(self, name: str) -> None:
        """Remove a field from the fieldset.
        Parameters:
            name: name of the field
        """
        if name not in self._fields:
            raise KeyError(f"Field '{name}' does not exist in Fieldset. Cannot remove.")
        del self._fields[name]

    def overwrite_field(self, name: str, field: Field) -> None:
        """Overwrite an existing field in the fieldset.
        Parameters:
            name: name of the field
            field: Field object
        """
        if name not in self._fields:
            raise KeyError(
                f"Field '{name}' does not exist in Fieldset. Cannot overwrite."
            )
        self._fields[name] = field

    def add_constant(self, name: str, value: Number) -> None:
        """Add a constant to the fieldset.
        Parameters:
            name: name of the constant
            value: value of the constant
        """
        if name in self._constants or name in self._fields:
            raise KeyError(
                f"'{name}' already exists in Fieldset. First remove it before adding a new one."
            )
        self._constants[name] = value

    def remove_constant(self, name: str) -> None:
        """Remove a constant from the fieldset.
        Parameters:
            name: name of the constant
        """
        if name not in self._constants:
            raise KeyError(
                f"Constant '{name}' does not exist in Fieldset. Cannot remove."
            )
        del self._constants[name]

    def overwrite_constant(self, name: str, value: Number) -> None:
        """Overwrite an existing constant in the fieldset.
        Parameters:
            name: name of the constant
            value: float value of the constant
        """
        if name not in self._constants:
            raise KeyError(
                f"Constant '{name}' does not exist in Fieldset. Cannot overwrite."
            )
        self._constants[name] = value

    def remove(self, name: str) -> None:
        """Remove a field or constant from the fieldset.
        Parameters:
            name: name of the field or constant
        """
        if name in self._fields:
            self.remove_field(name)
        elif name in self._constants:
            self.remove_constant(name)
        else:
            raise KeyError(f"'{name}' does not exist in Fieldset. Cannot remove.")

    def __getitem__(self, name: str) -> Field | Number:
        """Get a field or constant from the fieldset.
        Parameters:
            name: name of the field or constant
        Returns:
            Field object or float value of the constant
        """
        if name in self._fields:
            return self._fields[name]
        elif name in self._constants:
            return self._constants[name]
        else:
            raise KeyError(f"'{name}' does not exist in Fieldset.")

    def __contains__(self, name: str) -> bool:
        """Check if a field or constant exists in the fieldset.
        Parameters:
            name: name of the field or constant
        Returns:
            True if the field or constant exists, False otherwise
        """
        return name in self._fields or name in self._constants

    def keys(self) -> Iterable[str]:
        return itertools.chain(self._fields.keys(), self._constants.keys())

    def values(self) -> Iterable[Field | Number]:
        return itertools.chain(self._fields.values(), self._constants.values())

    def items(self) -> Iterable[tuple[str, Field | Number]]:
        return itertools.chain(self._fields.items(), self._constants.items())

    def __repr__(self) -> str:
        field_str = ", \n\t\t".join(
            f"{key} = {value}" for key, value in self._fields.items()
        )
        constants_str = ", \n\t\t".join(
            f"{key} = {value}" for key, value in self._constants.items()
        )
        return (
            f"FieldSet(\n\tt_size={self.t_size}, z_size={self.z_size}, y_size={self.y_size}, x_size={self.x_size},"
            + f"\n\ttime = array(shape={self.time.shape}, dtype={self.time.dtype}),"
            + f"\n\tfields = {{\n\t\t{field_str}\n\t}}"
            + f"\n\tconstants = {{\n\t\t{constants_str}\n\t}}\n)"
        )
