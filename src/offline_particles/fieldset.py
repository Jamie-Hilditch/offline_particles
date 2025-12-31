"""Submodule for Fieldset, a collection of fields from a simulation."""

import types
from typing import Any, ItemsView, KeysView, Mapping, ValuesView

import numpy as np

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
        constants: Mapping[str, Any] | None = None,
        zidx_bounds: tuple[float, float] | None = None,
        yidx_bounds: tuple[float, float] | None = None,
        xidx_bounds: tuple[float, float] | None = None,
        **fields: Field,
    ) -> None:
        super().__init__()
        # sizes of centered dimensions
        self._t_size = t_size
        self._z_size = z_size
        self._y_size = y_size
        self._x_size = x_size

        # set default index bounds if not provided
        if zidx_bounds is None:
            zidx_bounds = (0, z_size - 1)
        if yidx_bounds is None:
            yidx_bounds = (0, y_size - 1)
        if xidx_bounds is None:
            xidx_bounds = (0, x_size - 1)
        self._zidx_bounds = (np.float64(zidx_bounds[0]), np.float64(zidx_bounds[1]))
        self._yidx_bounds = (np.float64(yidx_bounds[0]), np.float64(yidx_bounds[1]))
        self._xidx_bounds = (np.float64(xidx_bounds[0]), np.float64(xidx_bounds[1]))

        self._fields: dict[str, Field] = {}
        self._constants: dict[str, np.number] = {}

        # add constants
        if constants is not None:
            for name, value in constants.items():
                self.add_constant(name, value)

        # add index bounds as constants
        self.add_constant("zidx_min", self._zidx_bounds[0])
        self.add_constant("zidx_max", self._zidx_bounds[1])
        self.add_constant("yidx_min", self._yidx_bounds[0])
        self.add_constant("yidx_max", self._yidx_bounds[1])
        self.add_constant("xidx_min", self._xidx_bounds[0])
        self.add_constant("xidx_max", self._xidx_bounds[1])

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
    def zidx_bounds(self) -> tuple[float, float]:
        """Bounds of the z index."""
        return self._zidx_bounds

    @property
    def yidx_bounds(self) -> tuple[float, float]:
        """Bounds of the y index."""
        return self._yidx_bounds

    @property
    def xidx_bounds(self) -> tuple[float, float]:
        """Bounds of the x index."""
        return self._xidx_bounds

    @property
    def zidx_min(self) -> float:
        """Minimum z index."""
        return self._zidx_bounds[0]

    @property
    def zidx_max(self) -> float:
        """Maximum z index."""
        return self._zidx_bounds[1]

    @property
    def yidx_min(self) -> float:
        """Minimum y index."""
        return self._yidx_bounds[0]

    @property
    def yidx_max(self) -> float:
        """Maximum y index."""
        return self._yidx_bounds[1]

    @property
    def xidx_min(self) -> float:
        """Minimum x index."""
        return self._xidx_bounds[0]

    @property
    def xidx_max(self) -> float:
        """Maximum x index."""
        return self._xidx_bounds[1]

    @property
    def simulation_shape(self) -> tuple[int, int, int, int]:
        """4D shape of the simulation assuming centered grids."""
        return (self._t_size, self._z_size, self._y_size, self._x_size)

    @property
    def fields(self) -> Mapping[str, Field]:
        """Dictionary of fields in the fieldset."""
        return types.MappingProxyType(self._fields)

    @property
    def constants(self) -> Mapping[str, np.number]:
        """Dictionary of constants in the fieldset."""
        return types.MappingProxyType(self._constants)

    def add_field(self, name: str, field: Field) -> None:
        """Add a field to the fieldset.
        Parameters:
            name: name of the field
            field: Field object
        """
        if name in self:
            raise KeyError(f"Field '{name}' already exists in Fieldset. First remove it before adding a new one.")
        try:
            field.validate_shape(self.simulation_shape)
        except ValueError as e:
            raise ValueError(f"Error validating shape of Field '{name}'.") from e
        self._fields[name] = field

    def add_constant(self, name: str, value: Any) -> None:
        """Convenience method for adding a constant field to the fieldset.
        Parameters:
            name: name of the constant
            value: value of the constant
        """
        if name in self._constants or name in self:
            raise KeyError(f"'{name}' already exists in Fieldset. First remove it before adding a new one.")
        self._constants[name] = _numpyify_constant(value)

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
        field_str = ", \n\t".join(f"{key} = {value}" for key, value in self._fields.items())
        return (
            f"Fieldset(\n\tt_size={self.t_size}, z_size={self.z_size}, y_size={self.y_size}, x_size={self.x_size},"
            + f"\n\t{constant_str}\n\t{field_str}\n)"
        )


def _numpyify_constant(value: Any) -> np.number:
    """Convert a value to a numpy scalar."""
    try:
        arr = np.asarray(value)
        if arr.size != 1:
            raise ValueError(f"Expected a single value, got array of size {arr.size}.")
        return arr.item()
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert value '{value}' to a numpy scalar.") from e
