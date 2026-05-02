"""Submodule for Fieldset, a collection of fields from a simulation."""

import types
from typing import Any, ItemsView, KeysView, Mapping, ValuesView

import numpy as np

from .fields import Field, SimulationSize


class Fieldset:
    """Class representing a collection of fields from a simulation.

    Can also hold associated constants.

    Parameters:
        t_size: size of the time dimension
        z_size: size of the centered z dimension
        y_size: size of the centered y dimension
        x_size: size of the centered x dimension
        fields: optional dictionary of fields to add to the fieldset
        constants: optional dictionary of constants to add to the fieldset
        zidx_bounds: optional bounds of the z index (default: (0, z_size - 1))
        yidx_bounds: optional bounds of the y index (default: (0, y_size - 1))
        xidx_bounds: optional bounds of the x index (default: (0, x_size - 1))
    """

    def __init__(
        self,
        t_size: int,
        z_size: int,
        y_size: int,
        x_size: int,
        *,
        fields: Mapping[str, Field] | None = None,
        constants: Mapping[str, Any] | None = None,
        zidx_bounds: tuple[float, float] | None = None,
        yidx_bounds: tuple[float, float] | None = None,
        xidx_bounds: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        # sizes of centered dimensions
        self._simulation_size = SimulationSize(t_size=t_size, z_size=z_size, y_size=y_size, x_size=x_size)

        # set default index bounds if not provided
        if zidx_bounds is None:
            zidx_bounds = (0, self._simulation_size.z - 1)
        if yidx_bounds is None:
            yidx_bounds = (0, self._simulation_size.y - 1)
        if xidx_bounds is None:
            xidx_bounds = (0, self._simulation_size.x - 1)
        self._zidx_bounds = (np.float64(zidx_bounds[0]), np.float64(zidx_bounds[1]))
        self._yidx_bounds = (np.float64(yidx_bounds[0]), np.float64(yidx_bounds[1]))
        self._xidx_bounds = (np.float64(xidx_bounds[0]), np.float64(xidx_bounds[1]))

        self._fields: dict[str, Field] = {}
        self._constants: dict[str, np.generic] = {}

        # add fields
        if fields is not None:
            for name, field in fields.items():
                self.add_field(name, field)

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

    @property
    def simulation_size(self) -> SimulationSize:
        """Simulation size as a SimulationSize named tuple."""
        return self._simulation_size

    @property
    def t_size(self) -> int:
        """Size of the time dimension."""
        return self.simulation_size.t

    @property
    def z_size(self) -> int:
        """Size of the centered z dimension."""
        return self.simulation_size.z

    @property
    def y_size(self) -> int:
        """Size of the centered y dimension."""
        return self.simulation_size.y

    @property
    def x_size(self) -> int:
        """Size of the centered x dimension."""
        return self.simulation_size.x

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
    def fields(self) -> Mapping[str, Field]:
        """Dictionary of fields in the fieldset."""
        return types.MappingProxyType(self._fields)

    @property
    def constants(self) -> Mapping[str, np.generic]:
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


def _numpyify_constant(value: Any) -> np.generic:
    """Convert a value to a numpy scalar."""
    try:
        arr = np.asarray(value)
        if arr.size != 1:
            raise ValueError(f"Expected a single value, got array of size {arr.size}.")
        return arr.item()
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert value '{value}' to a numpy scalar.") from e
