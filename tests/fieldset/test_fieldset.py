"""Tests for the Fieldset class."""

import numpy as np
import pytest

from offline_particles.fields import SimulationSize, StaticField
from offline_particles.fieldset import Fieldset


def _make_static_field(z: int, y: int, x: int) -> StaticField:
    """Create a minimal StaticField for testing.

    Parameters
    ----------
    z : int
        Size of the Z dimension.
    y : int
        Size of the Y dimension.
    x : int
        Size of the X dimension.

    Returns
    -------
    StaticField
        A StaticField with the specified dimensions, filled with zeros and
        with axes ("Z", "Y", "X") and staggers ("center", "center", "center").
    """
    data = np.zeros((z, y, x), dtype=np.float64)
    return StaticField.from_numpy(data, axes=("Z", "Y", "X"), staggers=("center", "center", "center"))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFieldsetConstruction:
    def test_basic_construction(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        assert fs.t_size == 10
        assert fs.z_size == 4
        assert fs.y_size == 5
        assert fs.x_size == 6

    def test_simulation_size(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        assert fs.simulation_size == SimulationSize(10, 4, 5, 6)

    def test_default_index_bounds(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        assert fs.zidx_bounds == (0, 3)
        assert fs.yidx_bounds == (0, 4)
        assert fs.xidx_bounds == (0, 5)

    def test_custom_index_bounds(self) -> None:
        fs = Fieldset(10, 4, 5, 6, zidx_bounds=(1, 2), yidx_bounds=(0.5, 3.5), xidx_bounds=(-0.5, 5.5))
        assert fs.zidx_bounds == (np.float64(1), np.float64(2))
        assert fs.yidx_bounds == (np.float64(0.5), np.float64(3.5))
        assert fs.xidx_bounds == (np.float64(-0.5), np.float64(5.5))

    def test_index_bounds_added_as_constants(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        assert "zidx_min" in fs.constants
        assert "zidx_max" in fs.constants
        assert "yidx_min" in fs.constants
        assert "yidx_max" in fs.constants
        assert "xidx_min" in fs.constants
        assert "xidx_max" in fs.constants

    def test_constants_kwarg(self) -> None:
        fs = Fieldset(10, 4, 5, 6, constants={"gravity": 9.81})
        assert "gravity" in fs.constants
        assert float(fs.constants["gravity"]) == pytest.approx(9.81)

    def test_fields_kwarg(self) -> None:
        field = _make_static_field(4, 5, 6)
        fs = Fieldset(10, 4, 5, 6, fields={"u": field})
        assert "u" in fs.fields


# ---------------------------------------------------------------------------
# add_field
# ---------------------------------------------------------------------------


class TestFieldsetAddField:
    def test_add_field(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        field = _make_static_field(4, 5, 6)
        fs.add_field("u", field)
        assert "u" in fs.fields

    def test_add_duplicate_field_raises(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        field = _make_static_field(4, 5, 6)
        fs.add_field("u", field)
        with pytest.raises(KeyError, match="u"):
            fs.add_field("u", field)

    def test_add_field_wrong_shape_raises(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        wrong_field = _make_static_field(3, 5, 6)  # wrong z dimension
        with pytest.raises(ValueError):
            fs.add_field("u", wrong_field)


# ---------------------------------------------------------------------------
# add_constant
# ---------------------------------------------------------------------------


class TestFieldsetAddConstant:
    def test_add_constant(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        fs.add_constant("rho0", 1025.0)
        assert "rho0" in fs.constants

    def test_add_duplicate_constant_raises(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        fs.add_constant("rho0", 1025.0)
        with pytest.raises(KeyError, match="rho0"):
            fs.add_constant("rho0", 1026.0)

    def test_add_constant_clashes_with_field_raises(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        field = _make_static_field(4, 5, 6)
        fs.add_field("u", field)
        # 'u' is in fields but not constants; add_constant checks both
        # Actually fieldset.__contains__ only checks _fields, not constants
        # But add_constant checks both self._constants and self (i.e. fields)
        with pytest.raises(KeyError, match="u"):
            fs.add_constant("u", 1.0)

    def test_add_constant_non_scalar_raises(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        with pytest.raises(ValueError):
            fs.add_constant("arr", [1.0, 2.0])


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


class TestFieldsetRemove:
    def test_remove_field(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        field = _make_static_field(4, 5, 6)
        fs.add_field("u", field)
        fs.remove("u")
        assert "u" not in fs.fields

    def test_remove_constant(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        fs.add_constant("rho0", 1025.0)
        fs.remove("rho0")
        assert "rho0" not in fs.constants

    def test_remove_missing_raises(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        with pytest.raises(KeyError, match="nonexistent"):
            fs.remove("nonexistent")


# ---------------------------------------------------------------------------
# __contains__ and __getitem__
# ---------------------------------------------------------------------------


class TestFieldsetContainsAndGetitem:
    def test_contains_added_field(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        field = _make_static_field(4, 5, 6)
        fs.add_field("u", field)
        assert "u" in fs

    def test_does_not_contain_missing(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        assert "u" not in fs

    def test_getitem_returns_field(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        field = _make_static_field(4, 5, 6)
        fs.add_field("u", field)
        assert fs["u"] is field

    def test_getitem_missing_raises(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        with pytest.raises(KeyError, match="u"):
            _ = fs["u"]


# ---------------------------------------------------------------------------
# keys, values, items
# ---------------------------------------------------------------------------


class TestFieldsetMapping:
    def test_keys(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        field = _make_static_field(4, 5, 6)
        fs.add_field("u", field)
        assert "u" in fs

    def test_values(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        field = _make_static_field(4, 5, 6)
        fs.add_field("u", field)
        assert field in fs.values()

    def test_items(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        field = _make_static_field(4, 5, 6)
        fs.add_field("u", field)
        assert ("u", field) in fs.items()


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


class TestFieldsetRepr:
    def test_repr_contains_sizes(self) -> None:
        fs = Fieldset(10, 4, 5, 6)
        r = repr(fs)
        assert "t_size=10" in r
        assert "z_size=4" in r
        assert "y_size=5" in r
        assert "x_size=6" in r
