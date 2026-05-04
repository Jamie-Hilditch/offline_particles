"""Tests for Fieldset.from_xarray."""

import numpy as np
import pytest
import xarray as xr

from offline_particles.fields import StaticField, TimeDependentField
from offline_particles.fieldset import Fieldset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    t: int = 3,
    z: int = 4,
    y: int = 5,
    x: int = 6,
    time_dim: str = "time",
    z_dim: str = "z",
    y_dim: str = "y",
    x_dim: str = "x",
) -> xr.Dataset:
    """Create a minimal xarray Dataset with one static and one time-dependent field."""
    static = xr.DataArray(np.ones((z, y, x), dtype=np.float64), dims=[z_dim, y_dim, x_dim])
    timedep = xr.DataArray(np.ones((t, z, y, x), dtype=np.float64), dims=[time_dim, z_dim, y_dim, x_dim])
    return xr.Dataset({"static": static, "timedep": timedep})


_DIMS = {"z": ("Z", "center"), "y": ("Y", "center"), "x": ("X", "center")}


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestFieldsetFromXarrayBasic:
    def test_returns_fieldset(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, "time", _DIMS)
        assert isinstance(fs, Fieldset)

    def test_sizes_inferred_from_dataset(self) -> None:
        ds = _make_dataset(t=3, z=4, y=5, x=6)
        fs = Fieldset.from_xarray(ds, "time", _DIMS)
        assert fs.t_size == 3
        assert fs.z_size == 4
        assert fs.y_size == 5
        assert fs.x_size == 6

    def test_static_field_created(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, "time", _DIMS)
        assert "static" in fs.fields
        assert isinstance(fs["static"], StaticField)

    def test_time_dependent_field_created(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, "time", _DIMS)
        assert "timedep" in fs.fields
        assert isinstance(fs["timedep"], TimeDependentField)

    def test_static_only_dataset(self) -> None:
        """Dataset with no time-dependent variables (time dim still required as a coordinate)."""
        ds = xr.Dataset(
            {
                "u": xr.DataArray(np.zeros((4, 5, 6), dtype=np.float64), dims=["z", "y", "x"]),
                "v": xr.DataArray(np.ones((4, 5, 6), dtype=np.float64), dims=["z", "y", "x"]),
            },
            coords={"time": np.arange(3)},
        )
        fs = Fieldset.from_xarray(ds, "time", _DIMS)
        assert isinstance(fs["u"], StaticField)
        assert isinstance(fs["v"], StaticField)

    def test_time_dependent_only_dataset(self) -> None:
        """Dataset with only time-dependent variables."""
        ds = xr.Dataset(
            {
                "u": xr.DataArray(np.zeros((3, 4, 5, 6), dtype=np.float64), dims=["time", "z", "y", "x"]),
            }
        )
        fs = Fieldset.from_xarray(ds, "time", _DIMS)
        assert isinstance(fs["u"], TimeDependentField)

    def test_default_index_bounds(self) -> None:
        ds = _make_dataset(z=4, y=5, x=6)
        fs = Fieldset.from_xarray(ds, "time", _DIMS)
        assert fs.zidx_bounds == (0, 3)
        assert fs.yidx_bounds == (0, 4)
        assert fs.xidx_bounds == (0, 5)

    def test_custom_index_bounds(self) -> None:
        ds = _make_dataset(z=4, y=5, x=6)
        fs = Fieldset.from_xarray(
            ds,
            "time",
            _DIMS,
            zidx_bounds=(1, 2),
            yidx_bounds=(0.5, 3.5),
            xidx_bounds=(-0.5, 5.5),
        )
        assert fs.zidx_bounds == (np.float64(1), np.float64(2))
        assert fs.yidx_bounds == (np.float64(0.5), np.float64(3.5))
        assert fs.xidx_bounds == (np.float64(-0.5), np.float64(5.5))


# ---------------------------------------------------------------------------
# Override sizes via z_size / y_size / x_size
# ---------------------------------------------------------------------------


class TestFieldsetFromXarrayOverrideSizes:
    def test_z_size_override(self) -> None:
        """z_size can be provided explicitly; inferred value must match dataset."""
        ds = _make_dataset(z=4, y=5, x=6)
        fs = Fieldset.from_xarray(ds, "time", _DIMS, z_size=4)
        assert fs.z_size == 4

    def test_all_sizes_overridden(self) -> None:
        ds = _make_dataset(t=3, z=4, y=5, x=6)
        fs = Fieldset.from_xarray(ds, "time", _DIMS, z_size=4, y_size=5, x_size=6)
        assert fs.z_size == 4
        assert fs.y_size == 5
        assert fs.x_size == 6

    def test_size_required_when_centered_dim_absent(self) -> None:
        """When a centered dimension is not in dims, its size must be provided."""
        # 2D dataset without z
        ds = xr.Dataset({"u": xr.DataArray(np.ones((3, 5, 6)), dims=["time", "y", "x"])})
        dims_2d = {"y": ("Y", "center"), "x": ("X", "center")}
        # z_size must be provided
        with pytest.raises(ValueError, match="centered z"):
            Fieldset.from_xarray(ds, "time", dims_2d)

    def test_y_size_required_when_y_absent(self) -> None:
        ds = xr.Dataset({"u": xr.DataArray(np.ones((3, 4, 6)), dims=["time", "z", "x"])})
        dims_no_y = {"z": ("Z", "center"), "x": ("X", "center")}
        with pytest.raises(ValueError, match="centered y"):
            Fieldset.from_xarray(ds, "time", dims_no_y)

    def test_x_size_required_when_x_absent(self) -> None:
        ds = xr.Dataset({"u": xr.DataArray(np.ones((3, 4, 5)), dims=["time", "z", "y"])})
        dims_no_x = {"z": ("Z", "center"), "y": ("Y", "center")}
        with pytest.raises(ValueError, match="centered x"):
            Fieldset.from_xarray(ds, "time", dims_no_x)

    def test_size_provided_for_absent_centered_dim(self) -> None:
        """Providing size for an absent centered dim should work."""
        ds = xr.Dataset({"u": xr.DataArray(np.ones((3, 5, 6)), dims=["time", "y", "x"])})
        dims_2d = {"y": ("Y", "center"), "x": ("X", "center")}
        fs = Fieldset.from_xarray(ds, "time", dims_2d, z_size=4, y_size=5, x_size=6)
        assert fs.z_size == 4
        assert fs.y_size == 5
        assert fs.x_size == 6


# ---------------------------------------------------------------------------
# Staggered dimensions
# ---------------------------------------------------------------------------


class TestFieldsetFromXarrayStaggered:
    def test_outer_stagger_z(self) -> None:
        """Outer-staggered z dimension has size z_size + 1."""
        t, z, y, x = 3, 4, 5, 6
        ds = xr.Dataset(
            {
                "w": xr.DataArray(np.ones((t, z + 1, y, x), dtype=np.float64), dims=["time", "zw", "y", "x"]),
            }
        )
        dims = {"zw": ("Z", "outer"), "y": ("Y", "center"), "x": ("X", "center")}
        fs = Fieldset.from_xarray(ds, "time", dims, z_size=z)
        assert fs.z_size == z
        assert isinstance(fs["w"], TimeDependentField)

    def test_inner_stagger_z(self) -> None:
        """Inner-staggered z dimension has size z_size - 1."""
        t, z, y, x = 3, 4, 5, 6
        ds = xr.Dataset(
            {
                "inner": xr.DataArray(np.ones((t, z - 1, y, x), dtype=np.float64), dims=["time", "zi", "y", "x"]),
            }
        )
        dims = {"zi": ("Z", "inner"), "y": ("Y", "center"), "x": ("X", "center")}
        fs = Fieldset.from_xarray(ds, "time", dims, z_size=z)
        assert fs.z_size == z

    def test_stagger_size_mismatch_raises(self) -> None:
        """If a staggered dimension size doesn't match expectation, raise ValueError."""
        t, z, y, x = 3, 4, 5, 6
        # Claim outer stagger (expects z+1=5) but use z=4
        ds = xr.Dataset(
            {
                "w": xr.DataArray(np.ones((t, z, y, x), dtype=np.float64), dims=["time", "zw", "y", "x"]),
            }
        )
        dims = {"zw": ("Z", "outer"), "y": ("Y", "center"), "x": ("X", "center")}
        with pytest.raises(ValueError, match="expected"):
            Fieldset.from_xarray(ds, "time", dims, z_size=z)


# ---------------------------------------------------------------------------
# Alias and string dim names
# ---------------------------------------------------------------------------


class TestFieldsetFromXarrayAliases:
    def test_axis_aliases(self) -> None:
        """String aliases like 'DEPTH', 'LATITUDE', 'LON' are accepted."""
        ds = _make_dataset()
        dims = {"z": ("DEPTH", "center"), "y": ("LATITUDE", "center"), "x": ("LON", "center")}
        fs = Fieldset.from_xarray(ds, "time", dims)
        assert isinstance(fs, Fieldset)

    def test_stagger_strings(self) -> None:
        """Stagger values given as strings are accepted."""
        ds = _make_dataset()
        dims = {"z": ("Z", "center"), "y": ("Y", "center"), "x": ("X", "center")}
        fs = Fieldset.from_xarray(ds, "time", dims)
        assert isinstance(fs, Fieldset)

    def test_custom_dim_names(self) -> None:
        """Non-default dimension names in the dataset are handled correctly."""
        ds = xr.Dataset(
            {
                "u": xr.DataArray(np.ones((3, 4, 5, 6)), dims=["T", "depth", "lat", "lon"]),
            }
        )
        dims = {"depth": ("Z", "center"), "lat": ("Y", "center"), "lon": ("X", "center")}
        fs = Fieldset.from_xarray(ds, "T", dims)
        assert fs.t_size == 3
        assert fs.z_size == 4
        assert fs.y_size == 5
        assert fs.x_size == 6


# ---------------------------------------------------------------------------
# Extra / droppable dimensions
# ---------------------------------------------------------------------------


class TestFieldsetFromXarrayDropDims:
    def test_extra_dims_dropped(self) -> None:
        """Dimensions not in time_dim or dims are dropped along with their variables."""
        ds = xr.Dataset(
            {
                "u": xr.DataArray(np.ones((3, 4, 5, 6)), dims=["time", "z", "y", "x"]),
                "extra": xr.DataArray(np.ones((7,)), dims=["extra_dim"]),
            }
        )
        fs = Fieldset.from_xarray(ds, "time", _DIMS)
        # 'extra' variable only has 'extra_dim' which is dropped
        assert "extra" not in fs.fields


# ---------------------------------------------------------------------------
# include_coords flag
# ---------------------------------------------------------------------------


class TestFieldsetFromXarrayIncludeCoords:
    def test_coords_excluded_by_default(self) -> None:
        """Coordinates are not included in the fieldset when include_coords is False (default)."""
        ds = xr.Dataset(
            {"u": xr.DataArray(np.ones((3, 4, 5, 6)), dims=["time", "z", "y", "x"])},
            coords={"depth": xr.DataArray(np.linspace(0, 100, 4), dims=["z"])},
        )
        fs = Fieldset.from_xarray(ds, "time", _DIMS)
        assert "u" in fs.fields
        assert "depth" not in fs.fields

    def test_coords_included_when_flag_true(self) -> None:
        """Coordinates are added as fields when include_coords=True."""
        ds = xr.Dataset(
            {"u": xr.DataArray(np.ones((3, 4, 5, 6)), dims=["time", "z", "y", "x"])},
            coords={"depth": xr.DataArray(np.linspace(0, 100, 4), dims=["z"])},
        )
        fs = Fieldset.from_xarray(ds, "time", _DIMS, include_coords=True)
        assert "u" in fs.fields
        assert "depth" in fs.fields

    def test_coord_becomes_static_field(self) -> None:
        """A spatial-only coordinate becomes a StaticField."""
        ds = xr.Dataset(
            {"u": xr.DataArray(np.ones((3, 4, 5, 6)), dims=["time", "z", "y", "x"])},
            coords={"depth": xr.DataArray(np.linspace(0, 100, 4), dims=["z"])},
        )
        fs = Fieldset.from_xarray(ds, "time", _DIMS, include_coords=True)
        assert isinstance(fs["depth"], StaticField)

    def test_time_dependent_coord_becomes_time_dependent_field(self) -> None:
        """A coordinate with a time dimension becomes a TimeDependentField."""
        ds = xr.Dataset(
            {"u": xr.DataArray(np.ones((3, 4, 5, 6)), dims=["time", "z", "y", "x"])},
            coords={"w": xr.DataArray(np.zeros((3, 4, 5, 6)), dims=["time", "z", "y", "x"])},
        )
        fs = Fieldset.from_xarray(ds, "time", _DIMS, include_coords=True)
        assert isinstance(fs["w"], TimeDependentField)

    def test_data_vars_still_included_with_coords(self) -> None:
        """Data variables are included regardless of include_coords value."""
        ds = xr.Dataset(
            {"u": xr.DataArray(np.ones((3, 4, 5, 6)), dims=["time", "z", "y", "x"])},
            coords={"depth": xr.DataArray(np.linspace(0, 100, 4), dims=["z"])},
        )
        fs = Fieldset.from_xarray(ds, "time", _DIMS, include_coords=True)
        assert "u" in fs.fields
        assert isinstance(fs["u"], TimeDependentField)

    def test_scalar_coord_excluded(self) -> None:
        """A scalar (0-d) coordinate is skipped when include_coords=True because it has no spatial dimensions."""
        ds = xr.Dataset(
            {"u": xr.DataArray(np.ones((3, 4, 5, 6)), dims=["time", "z", "y", "x"])},
            coords={"scalar_coord": 42.0},
        )
        fs = Fieldset.from_xarray(ds, "time", _DIMS, include_coords=True)
        # scalar coordinate has no spatial dims → skipped
        assert "scalar_coord" not in fs.fields

    def test_time_only_coord_excluded(self) -> None:
        """A time-only coordinate is skipped when include_coords=True because it has no spatial dimensions."""
        ds = xr.Dataset(
            {"u": xr.DataArray(np.ones((3, 4, 5, 6)), dims=["time", "z", "y", "x"])},
            coords={"time": np.arange(3, dtype=np.float64)},
        )
        fs = Fieldset.from_xarray(ds, "time", _DIMS, include_coords=True)
        # time-only coordinate has no spatial dims → skipped
        assert "time" not in fs.fields


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestFieldsetFromXarrayErrors:
    def test_missing_time_dim_raises(self) -> None:
        ds = _make_dataset()
        with pytest.raises(ValueError, match="Time dimension 'missing_time'"):
            Fieldset.from_xarray(ds, "missing_time", _DIMS)

    def test_missing_spatial_dim_raises(self) -> None:
        ds = _make_dataset()
        dims_with_extra = {**_DIMS, "nonexistent": ("Z", "center")}
        with pytest.raises(ValueError, match="Spatial dimension 'nonexistent'"):
            Fieldset.from_xarray(ds, "time", dims_with_extra)

    def test_size_mismatch_raises(self) -> None:
        """z_size override that conflicts with dataset dimension raises ValueError."""
        ds = _make_dataset(z=4)
        # Override z_size to wrong value (3 instead of 4) with center stagger → expects 3, gets 4
        with pytest.raises(ValueError, match="expected"):
            Fieldset.from_xarray(ds, "time", _DIMS, z_size=3)
