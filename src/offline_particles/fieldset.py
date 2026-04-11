"""Submodule for Fieldset, a collection of fields from a simulation."""

import types
from typing import TYPE_CHECKING, Any, ItemsView, KeysView, Mapping, ValuesView

import numpy as np

from .fields import Field, field_from_dataarray
from .spatial_arrays import Dimension, Stagger

if TYPE_CHECKING:
    import xarray as xr


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
        self._constants: dict[str, np.generic] = {}

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

    @classmethod
    def from_xarray(
        cls,
        ds: "xr.Dataset",
        dim_map: Mapping[str, Dimension],
        *,
        t_size: int | None = None,
        z_size: int | None = None,
        y_size: int | None = None,
        x_size: int | None = None,
        constants: Mapping[str, Any] | None = None,
        zidx_bounds: tuple[float, float] | None = None,
        yidx_bounds: tuple[float, float] | None = None,
        xidx_bounds: tuple[float, float] | None = None,
    ) -> "Fieldset":
        """Build a :class:`Fieldset` from an :class:`xarray.Dataset`.

        Every data variable in the dataset is converted to a :class:`~offline_particles.fields.Field`
        using :func:`~offline_particles.fields.field_from_dataarray`.  The
        :class:`~offline_particles.spatial_arrays.Dimension` entries in *dim_map*
        declare how each dataset dimension maps to the simulation axes (T, Z, Y, X)
        and to a grid :class:`~offline_particles.spatial_arrays.Stagger`.

        Parameters
        ----------
        ds:
            The :class:`xarray.Dataset` to convert.
        dim_map:
            Mapping from dataset dimension names to
            :class:`~offline_particles.spatial_arrays.Dimension` values.  All
            dimensions that appear in any data variable of *ds* must be covered.
        t_size:
            Size of the time dimension.  Inferred from *ds* when ``None``.
        z_size:
            Size of the **centred** z dimension.  Inferred from *ds* when
            ``None``; if the mapped z dimension is staggered, the centred size
            is computed from the stagger relationship.
        y_size:
            Size of the **centred** y dimension.  Inferred from *ds* when ``None``.
        x_size:
            Size of the **centred** x dimension.  Inferred from *ds* when ``None``.
        constants:
            Optional extra constants to include in the :class:`Fieldset`.
        zidx_bounds, yidx_bounds, xidx_bounds:
            Custom index bounds passed through to
            :class:`Fieldset.__init__`.  Defaults are ``(0, size - 1)`` for
            each spatial direction.

        Returns
        -------
        Fieldset
            A :class:`Fieldset` containing one :class:`~offline_particles.fields.Field`
            per data variable in *ds*.

        Raises
        ------
        TypeError
            If *ds* is not an :class:`xarray.Dataset`.
        ValueError
            If a required dimension size cannot be inferred and was not supplied,
            or if *dim_map* is inconsistent (e.g. two dimensions map to the
            same spatial direction in the same variable).
        """
        import xarray as xr

        if not isinstance(ds, xr.Dataset):
            raise TypeError(f"Expected an xarray Dataset, got {type(ds).__name__!r}.")

        # Infer dimension sizes from the dataset where not explicitly provided.
        inferred_t, inferred_z, inferred_y, inferred_x = _infer_sizes_from_dataset(ds, dim_map)

        resolved_t = t_size if t_size is not None else inferred_t
        resolved_z = z_size if z_size is not None else inferred_z
        resolved_y = y_size if y_size is not None else inferred_y
        resolved_x = x_size if x_size is not None else inferred_x

        missing = [
            name
            for name, val in [("t_size", resolved_t), ("z_size", resolved_z), ("y_size", resolved_y), ("x_size", resolved_x)]
            if val is None
        ]
        if missing:
            raise ValueError(
                f"Could not infer {missing!r} from the dataset and dim_map. "
                "Please supply the missing size(s) explicitly."
            )

        assert resolved_t is not None and resolved_z is not None
        assert resolved_y is not None and resolved_x is not None

        fieldset = cls(
            resolved_t,
            resolved_z,
            resolved_y,
            resolved_x,
            constants=constants,
            zidx_bounds=zidx_bounds,
            yidx_bounds=yidx_bounds,
            xidx_bounds=xidx_bounds,
        )

        for var_name, data_array in ds.data_vars.items():
            field = field_from_dataarray(data_array, dim_map)
            fieldset.add_field(str(var_name), field)

        return fieldset


def _numpyify_constant(value: Any) -> np.generic:
    """Convert a value to a numpy scalar."""
    try:
        arr = np.asarray(value)
        if arr.size != 1:
            raise ValueError(f"Expected a single value, got array of size {arr.size}.")
        return arr.item()
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert value '{value}' to a numpy scalar.") from e


def _staggered_to_centered_size(actual_size: int, stagger: Stagger) -> int:
    """Return the centred-grid size that corresponds to *actual_size* on *stagger*."""
    match stagger:
        case Stagger.CENTER | Stagger.LEFT | Stagger.RIGHT:
            return actual_size
        case Stagger.INNER:
            return actual_size + 1
        case Stagger.OUTER:
            return actual_size - 1
        case _:
            # INVARIANT or unknown – caller should not reach here
            raise ValueError(f"Cannot compute centred size for stagger {stagger!r}.")


def _infer_sizes_from_dataset(
    ds: "xr.Dataset",
    dim_map: Mapping[str, Dimension],
) -> tuple[int | None, int | None, int | None, int | None]:
    """Infer t_size, z_size, y_size, x_size from *ds* and *dim_map*.

    Returns ``None`` for any direction not represented in *dim_map*.
    """
    t_size: int | None = None
    spatial_sizes: dict[str, int | None] = {"Z": None, "Y": None, "X": None}

    for dim_name, dimension in dim_map.items():
        if dim_name not in ds.dims:
            continue
        actual_size: int = ds.sizes[dim_name]

        if dimension.is_time:
            if t_size is not None and t_size != actual_size:
                raise ValueError(
                    f"Conflicting time dimension sizes: {t_size} vs {actual_size} "
                    f"(from dimension {dim_name!r})."
                )
            t_size = actual_size
        else:
            stagger: Stagger | None = dimension.stagger  # type: ignore[assignment]
            if stagger is None or stagger is Stagger.INVARIANT:
                continue
            centered = _staggered_to_centered_size(actual_size, stagger)
            direction: str = dimension.direction
            existing = spatial_sizes[direction]
            if existing is not None and existing != centered:
                raise ValueError(
                    f"Inconsistent centred sizes for direction '{direction}': "
                    f"{existing} vs {centered} (from dimension {dim_name!r})."
                )
            spatial_sizes[direction] = centered

    return t_size, spatial_sizes["Z"], spatial_sizes["Y"], spatial_sizes["X"]
