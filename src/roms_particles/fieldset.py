"""Submodule for Fieldset, a collection of fields from a simulation."""

from collections.abc import ItemsView, KeysView, ValuesView
from numbers import Number

from .fields import ConstantField, Field
from .kernel_data import KernelDataSource


class Fieldset(KernelDataSource):
    def __init__(
        self,
        t_size: int,
        z_size: int,
        y_size: int,
        x_size: int,
        **kwargs: Field,
    ) -> None:
        # sizes of centered dimensions
        self._t_size = t_size
        self._z_size = z_size
        self._y_size = y_size
        self._x_size = x_size

        self._fields: dict[str, Field] = {}

        # add fields and constants
        for name, field in kwargs.items():
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
    def fields(self) -> dict[str, Field]:
        """Dictionary of fields in the fieldset."""
        return self._fields

    def add_field(self, name: str, field: Field) -> None:
        """Add a field to the fieldset.
        Parameters:
            name: name of the field
            field: Field object
        """
        if name in self._fields:
            raise KeyError(
                f"'{name}' already exists in Fieldset. First remove it before adding a new one."
            )
        try:
            field.validate_shape(self.simulation_shape)
        except ValueError as e:
            raise ValueError(f"Error validating shape of Field '{name}'.") from e
        self._fields[name] = field
        self.register_kernel_data_function(name, field.get_kernel_data)

    def remove_field(self, name: str) -> None:
        """Remove a field from the fieldset.
        Parameters:
            name: name of the field
        """
        if name not in self._fields:
            raise KeyError(f"Field '{name}' does not exist in Fieldset. Cannot remove.")
        self.deregister_kernel_data_function(name)
        del self._fields[name]

    def add_constant(self, name: str, value: Number) -> None:
        """Convenience method for adding a constant field to the fieldset.
        Parameters:
            name: name of the constant
            value: value of the constant
        """
        self.add_field(name, ConstantField(value))

    def __getitem__(self, name: str) -> Field:
        """Get a field from the fieldset.
        Parameters:
            name: name of the field or constant
        Returns:
            Field object or float value of the constant
        """
        if name not in self._fields:
            raise KeyError(f"'{name}' does not exist in Fieldset.")
        return self._fields[name]

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
        field_str = ", \n\t".join(
            f"{key} = {value}" for key, value in self._fields.items()
        )
        return (
            f"FieldSet(\n\tt_size={self.t_size}, z_size={self.z_size}, y_size={self.y_size}, x_size={self.x_size},"
            + f"\n\t{field_str}\n)"
        )
