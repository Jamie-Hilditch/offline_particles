"""Submodule for Fieldset, a collection of fields from a simulation."""

import types
from numbers import Number
from typing import ItemsView, KeysView, Mapping, ValuesView

from .fields import Field


class Fieldset:
    """Class representing a collection of fields from a simulation.

    Can also hold associated constants.

    Parameters:
        t_size: size of the time dimension
        z_size: size of the centered z dimension
        y_size: size of the centered y dimension
        x_size: size of the centered x dimension
        constants: optional keyword argument, dictionary of constants to add to the fieldset
        fields: fields to add to the fieldset as keyword arguments
    """

    def __init__(
        self,
        t_size: int,
        z_size: int,
        y_size: int,
        x_size: int,
        *,
        constants: dict[str, Number] | None = None,
        **fields: Field,
    ) -> None:
        super().__init__()
        # sizes of centered dimensions
        self._t_size = t_size
        self._z_size = z_size
        self._y_size = y_size
        self._x_size = x_size

        self._fields: dict[str, Field] = {}
        self._constants: dict[str, Number] = {}

        # add constants
        if constants is not None:
            for name, value in constants.items():
                self.add_constant(name, value)

        # add fields
        for name, field in fields.items():
            self.add_field(name, field)

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
    def fields(self) -> Mapping[str, Field]:
        """Dictionary of fields in the fieldset."""
        return types.MappingProxyType(self._fields)

    @property
    def constants(self) -> Mapping[str, Number]:
        """Dictionary of constants in the fieldset."""
        return types.MappingProxyType(self._constants)

    def add_field(self, name: str, field: Field) -> None:
        """Add a field to the fieldset.
        Parameters:
            name: name of the field
            field: Field object
        """
        if name in self:
            raise KeyError(
                f"Field '{name}' already exists in Fieldset. First remove it before adding a new one."
            )
        try:
            field.validate_shape(self.simulation_shape)
        except ValueError as e:
            raise ValueError(f"Error validating shape of Field '{name}'.") from e
        self._fields[name] = field

    def add_constant(self, name: str, value: Number) -> None:
        """Convenience method for adding a constant field to the fieldset.
        Parameters:
            name: name of the constant
            value: value of the constant
        """
        if name in self._constants or name in self:
            raise KeyError(
                f"'{name}' already exists in Fieldset. First remove it before adding a new one."
            )
        self._constants[name] = value

    def remove(self, name: str) -> None:
        """Remove a field or constant from the fieldset.
        Parameters:
            name: name of the field
        """
        if name in self._constants:
            del self._constants[name]
            return
        if name in self._fields:
            del self._fields[name]
            return 
        raise KeyError(f"Field '{name}' does not exist in Fieldset. Cannot remove.")

        del self._fields[name]


    def __getitem__(self, name: str) -> Field:
        """Get a field from the fieldset.
        Parameters:
            name: name of the field or constant
        Returns:
            Field object or float value of the constant
        """
        if name in self._fields:
            return self._fields[name]
        raise KeyError(f"Field '{name}' does not exist in Fieldset.")
        

    def __contains__(self, name: str) -> bool:
        """Check if a field exists in the fieldset.
        Parameters:
            name: name of the field
        Returns:
            True if the field exists, False otherwise
        """
        return name in self._fields

    def keys(self) -> KeysView[str]:
        return self._fields.keys()

    def values(self) -> ValuesView[Field]:
        return self._fields.values()

    def items(self) -> ItemsView[str, Field]:
        return self._fields.items()

    def __repr__(self) -> str:
        constant_str = f"constants={self._constants}, "
        field_str = ", \n\t".join(
            f"{key} = {value}" for key, value in self._fields.items()
        )
        return (
            f"Fieldset(\n\tt_size={self.t_size}, z_size={self.z_size}, y_size={self.y_size}, x_size={self.x_size},"
            + f"\n\t{constant_str}\n\t{field_str}\n)"
        )
