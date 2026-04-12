"""Tests for StaticField.from_xarray and TimeDependentField.from_xarray."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from offline_particles.fields import StaticField, TimeDependentField


class TestStaticFieldFromXarray:
    def test_3d_numpy_backed(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5), dtype=np.float64), dims=["z", "y", "x"])
        field = StaticField.from_xarray(data, z=("Z", "center"), y=("Y", "center"), x=("X", "center"))
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (3, 4, 5)
        assert field.z_stagger.value == "center"
        assert field.y_stagger.value == "center"
        assert field.x_stagger.value == "center"

    def test_3d_positional_mapping(self) -> None:
        """Should work when dims is passed as a positional mapping instead of kwargs."""
        data = xr.DataArray(np.ones((3, 4, 5), dtype=np.float64), dims=["z", "y", "x"])
        dims = {"z": ("Z", "center"), "y": ("Y", "center"), "x": ("X", "center")}
        field = StaticField.from_xarray(data, dims)
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (3, 4, 5)

    def test_positional_mapping_supports_non_identifier_dim_names(self) -> None:
        """Dimension names that are not valid Python identifiers require the mapping style."""
        data = xr.DataArray(np.ones((3, 4), dtype=np.float64), dims=["y-coord", "x-coord"])
        dims = {"y-coord": ("Y", "center"), "x-coord": ("X", "center")}
        field = StaticField.from_xarray(data, dims)
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (3, 4)

    def test_error_both_dims_and_kwargs(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5), dtype=np.float64), dims=["z", "y", "x"])
        with pytest.raises(TypeError, match="cannot specify both 'dims' and keyword arguments"):
            StaticField.from_xarray(data, {"z": ("Z", "center")}, y=("Y", "center"), x=("X", "center"))

    def test_2d_numpy_backed_no_z(self) -> None:
        """Creating from a 2D (y, x) DataArray should succeed (Z axis absent)."""
        data = xr.DataArray(np.ones((4, 5), dtype=np.float64), dims=["y", "x"])
        field = StaticField.from_xarray(data, y=("Y", "center"), x=("X", "center"))
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (4, 5)
        assert field.z_stagger.is_invariant

    def test_1d_numpy_backed_x_only(self) -> None:
        """Creating from a 1D (x) DataArray should succeed (Z and Y axes absent)."""
        data = xr.DataArray(np.ones((5,), dtype=np.float64), dims=["x"])
        field = StaticField.from_xarray(data, x=("X", "center"))
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (5,)
        assert field.z_stagger.is_invariant
        assert field.y_stagger.is_invariant

    def test_invariant_z_dimension_squeezed(self) -> None:
        """A singleton z dimension marked as invariant should be squeezed out."""
        data = xr.DataArray(np.ones((1, 4, 5), dtype=np.float64), dims=["z", "y", "x"])
        field = StaticField.from_xarray(data, z=("Z", "invariant"), y=("Y", "center"), x=("X", "center"))
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (4, 5)
        assert field.z_stagger.is_invariant

    def test_3d_dask_backed(self) -> None:
        data = xr.DataArray(da.ones((3, 4, 5), chunks=(3, 4, 5), dtype=np.float64), dims=["z", "y", "x"])
        field = StaticField.from_xarray(data, z=("Z", "center"), y=("Y", "center"), x=("X", "center"))
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (3, 4, 5)

    def test_attrs_preserved(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["z", "y", "x"], attrs={"units": "m/s"})
        field = StaticField.from_xarray(data, z=("Z", "center"), y=("Y", "center"), x=("X", "center"))
        assert field.attrs == {"units": "m/s"}

    def test_dim_order_independent(self) -> None:
        """DataArray with (x, y, z) order should be transposed correctly."""
        data = xr.DataArray(np.arange(60, dtype=np.float64).reshape(5, 4, 3), dims=["x", "y", "z"])
        field = StaticField.from_xarray(data, z=("Z", "center"), y=("Y", "center"), x=("X", "center"))
        assert isinstance(field, StaticField)
        # after transpose to (z, y, x) shape should be (3, 4, 5)
        assert field.spatial_shape == (3, 4, 5)

    def test_aliases_resolve_correctly(self) -> None:
        # "DEPTH" → ArrayAxis.Z, "LATITUDE" → ArrayAxis.Y, "LON" → ArrayAxis.X
        data = xr.DataArray(np.ones((3, 4, 5), dtype=np.float64), dims=["depth", "lat", "lon"])
        field = StaticField.from_xarray(data, depth=("DEPTH", "center"), lat=("LATITUDE", "center"), lon=("LON", "center"))
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (3, 4, 5)
        assert field.z_stagger.value == "center"  # DEPTH → Z
        assert field.y_stagger.value == "center"  # LATITUDE → Y
        assert field.x_stagger.value == "center"  # LON → X

    def test_validation_error_missing_dim(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["z", "y", "x"])
        with pytest.raises(ValueError, match="Mismatch"):
            StaticField.from_xarray(data, z=("Z", "center"), y=("Y", "center"))  # missing x

    def test_validation_error_extra_dim(self) -> None:
        data = xr.DataArray(np.ones((4, 5)), dims=["y", "x"])
        with pytest.raises(ValueError, match="Mismatch"):
            StaticField.from_xarray(data, z=("Z", "center"), y=("Y", "center"), x=("X", "center"))  # extra z

    def test_validation_error_duplicate_axis(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["z", "y", "x"])
        with pytest.raises(ValueError, match="Multiple dimensions mapped"):
            StaticField.from_xarray(data, z=("Z", "center"), y=("Z", "center"), x=("X", "center"))


class TestTimeDependentFieldFromXarray:
    def test_4d_numpy_backed(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5, 6), dtype=np.float64), dims=["t", "z", "y", "x"])
        field = TimeDependentField.from_xarray(data, "t", z=("Z", "center"), y=("Y", "center"), x=("X", "center"))
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (4, 5, 6)

    def test_4d_positional_mapping(self) -> None:
        """Should work when dims is passed as a positional mapping instead of kwargs."""
        data = xr.DataArray(np.ones((3, 4, 5, 6), dtype=np.float64), dims=["t", "z", "y", "x"])
        dims = {"z": ("Z", "center"), "y": ("Y", "center"), "x": ("X", "center")}
        field = TimeDependentField.from_xarray(data, "t", dims)
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (4, 5, 6)

    def test_positional_mapping_supports_non_identifier_dim_names(self) -> None:
        """Dimension names that are not valid Python identifiers require the mapping style."""
        data = xr.DataArray(np.ones((3, 4, 5), dtype=np.float64), dims=["t", "y-coord", "x-coord"])
        dims = {"y-coord": ("Y", "center"), "x-coord": ("X", "center")}
        field = TimeDependentField.from_xarray(data, "t", dims)
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (4, 5)

    def test_error_both_dims_and_kwargs(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5, 6), dtype=np.float64), dims=["t", "z", "y", "x"])
        with pytest.raises(TypeError, match="cannot specify both 'dims' and keyword arguments"):
            TimeDependentField.from_xarray(
                data, "t", {"z": ("Z", "center")}, y=("Y", "center"), x=("X", "center")
            )

    def test_3d_no_z(self) -> None:
        """Creating from (t, y, x) DataArray should succeed (Z axis absent)."""
        data = xr.DataArray(np.ones((3, 4, 5), dtype=np.float64), dims=["t", "y", "x"])
        field = TimeDependentField.from_xarray(data, "t", y=("Y", "center"), x=("X", "center"))
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (4, 5)
        assert field.z_stagger.is_invariant

    def test_2d_time_plus_x(self) -> None:
        """Creating from (t, x) DataArray should succeed."""
        data = xr.DataArray(np.ones((3, 5), dtype=np.float64), dims=["t", "x"])
        field = TimeDependentField.from_xarray(data, "t", x=("X", "center"))
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (5,)
        assert field.z_stagger.is_invariant
        assert field.y_stagger.is_invariant

    def test_invariant_z_dimension_squeezed(self) -> None:
        """A singleton z dimension marked as invariant should be squeezed out."""
        data = xr.DataArray(np.ones((3, 1, 4, 5), dtype=np.float64), dims=["t", "z", "y", "x"])
        field = TimeDependentField.from_xarray(data, "t", z=("Z", "invariant"), y=("Y", "center"), x=("X", "center"))
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (4, 5)
        assert field.z_stagger.is_invariant

    def test_4d_dask_backed(self) -> None:
        data = xr.DataArray(
            da.ones((3, 4, 5, 6), chunks=(1, 4, 5, 6), dtype=np.float64), dims=["t", "z", "y", "x"]
        )
        field = TimeDependentField.from_xarray(data, "t", z=("Z", "center"), y=("Y", "center"), x=("X", "center"))
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (4, 5, 6)

    def test_attrs_preserved(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["t", "y", "x"], attrs={"units": "m/s"})
        field = TimeDependentField.from_xarray(data, "t", y=("Y", "center"), x=("X", "center"))
        assert field.attrs == {"units": "m/s"}

    def test_dim_order_independent(self) -> None:
        """DataArray with (z, t, x, y) order should be transposed correctly."""
        data = xr.DataArray(np.ones((4, 3, 6, 5), dtype=np.float64), dims=["z", "t", "x", "y"])
        field = TimeDependentField.from_xarray(data, "t", z=("Z", "center"), y=("Y", "center"), x=("X", "center"))
        assert isinstance(field, TimeDependentField)
        # after transpose to (t, z, y, x) shape should be (3, 4, 5, 6)
        assert field.data.shape == (3, 4, 5, 6)

    def test_validation_error_missing_time_dim(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["t", "y", "x"])
        with pytest.raises(ValueError, match="Time dimension 'time' not found"):
            TimeDependentField.from_xarray(data, "time", y=("Y", "center"), x=("X", "center"))

    def test_validation_error_missing_spatial_dim(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["t", "y", "x"])
        with pytest.raises(ValueError, match="Mismatch"):
            TimeDependentField.from_xarray(data, "t", y=("Y", "center"))  # missing x

    def test_validation_error_extra_spatial_dim(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["t", "y", "x"])
        with pytest.raises(ValueError, match="Mismatch"):
            TimeDependentField.from_xarray(data, "t", z=("Z", "center"), y=("Y", "center"), x=("X", "center"))

    def test_validation_error_duplicate_axis(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5, 6)), dims=["t", "z", "y", "x"])
        with pytest.raises(ValueError, match="Multiple dimensions mapped"):
            TimeDependentField.from_xarray(data, "t", z=("Z", "center"), y=("Z", "center"), x=("X", "center"))
