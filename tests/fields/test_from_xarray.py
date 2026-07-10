"""Tests for StaticField.from_xarray and TimeDependentField.from_xarray."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from offline_particles.fields import StaticField, TimeDependentField
from offline_particles.spatial_arrays import ArrayAxis, Stagger

STANDARD_DIMS = {"z": ("Z", "center"), "y": ("Y", "center"), "x": ("X", "center")}


class TestStaticFieldFromXarray:
    def test_3d_numpy_backed(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5), dtype=np.float64), dims=["z", "y", "x"])
        field = StaticField.from_xarray(data, STANDARD_DIMS)
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (3, 4, 5)
        assert field.staggers == (Stagger.CENTER, Stagger.CENTER, Stagger.CENTER)

    def test_3d_dims_as_variable(self) -> None:
        """Dims mapping can be passed as a variable."""
        data = xr.DataArray(np.ones((3, 4, 5), dtype=np.float64), dims=["z", "y", "x"])
        dims = STANDARD_DIMS
        field = StaticField.from_xarray(data, dims)
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (3, 4, 5)

    def test_non_identifier_dim_names(self) -> None:
        """Dimension names that are not valid Python identifiers are supported."""
        data = xr.DataArray(np.ones((3, 4), dtype=np.float64), dims=["y-coord", "x-coord"])
        dims = {"y-coord": ("Y", "center"), "x-coord": ("X", "center")}
        field = StaticField.from_xarray(data, dims)
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (3, 4)

    def test_2d_numpy_backed_no_z(self) -> None:
        """Creating from a 2D (y, x) DataArray should succeed (Z axis absent)."""
        data = xr.DataArray(np.ones((4, 5), dtype=np.float64), dims=["y", "x"])
        field = StaticField.from_xarray(data, {"y": ("Y", "center"), "x": ("X", "center")})
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (4, 5)
        assert ArrayAxis.Z not in field.axes

    def test_1d_numpy_backed_x_only(self) -> None:
        """Creating from a 1D (x) DataArray should succeed (Z and Y axes absent)."""
        data = xr.DataArray(np.ones((5,), dtype=np.float64), dims=["x"])
        field = StaticField.from_xarray(data, {"x": ("X", "center")})
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (5,)
        assert ArrayAxis.Z not in field.axes
        assert ArrayAxis.Y not in field.axes
        assert ArrayAxis.X in field.axes

    def test_3d_dask_backed(self) -> None:
        data = xr.DataArray(da.ones((3, 4, 5), chunks=(3, 4, 5), dtype=np.float64), dims=["z", "y", "x"])
        field = StaticField.from_xarray(data, STANDARD_DIMS)
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (3, 4, 5)

    def test_attrs_preserved(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["z", "y", "x"], attrs={"units": "m/s"})
        field = StaticField.from_xarray(data, STANDARD_DIMS)
        assert field.attrs == {"units": "m/s"}

    def test_dim_order_preserved(self) -> None:
        """Field preserves the dimension ordering of the input DataArray."""
        data = xr.DataArray(np.arange(60, dtype=np.float64).reshape(5, 4, 3), dims=["x", "y", "z"])
        field = StaticField.from_xarray(data, STANDARD_DIMS)
        assert isinstance(field, StaticField)
        # field preserves the dim ordering of the xarray DataArray (x, y, z) → shape (5, 4, 3)
        assert field.axes == (ArrayAxis.X, ArrayAxis.Y, ArrayAxis.Z)
        assert field.spatial_shape == (5, 4, 3)

    def test_aliases_resolve_correctly(self) -> None:
        # "DEPTH" → ArrayAxis.Z, "LATITUDE" → ArrayAxis.Y, "LON" → ArrayAxis.X
        data = xr.DataArray(np.ones((3, 4, 5), dtype=np.float64), dims=["depth", "lat", "lon"])
        field = StaticField.from_xarray(
            data, {"depth": ("DEPTH", "center"), "lat": ("LATITUDE", "center"), "lon": ("LON", "center")}
        )
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (3, 4, 5)
        assert ArrayAxis.Z in field.axes
        assert ArrayAxis.Y in field.axes
        assert ArrayAxis.X in field.axes

    def test_validation_error_missing_dim(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["z", "y", "x"])
        with pytest.raises(ValueError, match="Dimension 'x' in data is missing from dims mapping"):
            StaticField.from_xarray(data, {"z": ("Z", "center"), "y": ("Y", "center")})  # missing x

    def test_validation_error_extra_dim(self) -> None:
        data = xr.DataArray(np.ones((4, 5)), dims=["y", "x"])
        with pytest.raises(ValueError, match="Dimensions in dims mapping not found in data"):
            StaticField.from_xarray(data, STANDARD_DIMS)  # extra z

    def test_validation_error_duplicate_axis(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["z", "y", "x"])
        with pytest.raises(ValueError, match="Axes must be unique"):
            StaticField.from_xarray(
                data, {"z": ("Z", "center"), "y": ("Z", "center"), "x": ("X", "center")}
            )  # both z and y map to Z axis

    def test_ignore_missing_dims_true_skips_extra_dims_mapping(self) -> None:
        """When ignore_missing_dims=True, dims entries absent from the data are silently ignored."""
        data = xr.DataArray(np.ones((4, 5), dtype=np.float64), dims=["y", "x"])
        field = StaticField.from_xarray(
            data,
            STANDARD_DIMS,
            ignore_missing_dims=True,
        )
        assert isinstance(field, StaticField)
        assert field.spatial_shape == (4, 5)
        assert ArrayAxis.Z not in field.axes


class TestTimeDependentFieldFromXarray:
    def test_4d_numpy_backed(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5, 6), dtype=np.float64), dims=["t", "z", "y", "x"])
        field = TimeDependentField.from_xarray(data, "t", STANDARD_DIMS)
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (4, 5, 6)

    def test_4d_dims_as_variable(self) -> None:
        """Dims mapping can be passed as a variable."""
        data = xr.DataArray(np.ones((3, 4, 5, 6), dtype=np.float64), dims=["t", "z", "y", "x"])
        dims = STANDARD_DIMS
        field = TimeDependentField.from_xarray(data, "t", dims)
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (4, 5, 6)

    def test_non_identifier_dim_names(self) -> None:
        """Dimension names that are not valid Python identifiers are supported."""
        data = xr.DataArray(np.ones((3, 4, 5), dtype=np.float64), dims=["t", "y-coord", "x-coord"])
        dims = {"y-coord": ("Y", "center"), "x-coord": ("X", "center")}
        field = TimeDependentField.from_xarray(data, "t", dims)
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (4, 5)

    def test_3d_no_z(self) -> None:
        """Creating from (t, y, x) DataArray should succeed (Z axis absent)."""
        data = xr.DataArray(np.ones((3, 4, 5), dtype=np.float64), dims=["t", "y", "x"])
        field = TimeDependentField.from_xarray(data, "t", {"y": ("Y", "center"), "x": ("X", "center")})
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (4, 5)
        assert ArrayAxis.Z not in field.axes

    def test_2d_time_plus_x(self) -> None:
        """Creating from (t, x) DataArray should succeed."""
        data = xr.DataArray(np.ones((3, 5), dtype=np.float64), dims=["t", "x"])
        field = TimeDependentField.from_xarray(data, "t", {"x": ("X", "center")})
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (5,)
        assert ArrayAxis.Z not in field.axes
        assert ArrayAxis.Y not in field.axes
        assert ArrayAxis.X in field.axes

    def test_4d_dask_backed(self) -> None:
        data = xr.DataArray(da.ones((3, 4, 5, 6), chunks=(1, 4, 5, 6), dtype=np.float64), dims=["t", "z", "y", "x"])
        field = TimeDependentField.from_xarray(data, "t", STANDARD_DIMS)
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (4, 5, 6)

    def test_attrs_preserved(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["t", "y", "x"], attrs={"units": "m/s"})
        field = TimeDependentField.from_xarray(data, "t", {"y": ("Y", "center"), "x": ("X", "center")})
        assert field.attrs == {"units": "m/s"}

    def test_time_dim_moved_to_front(self) -> None:
        """DataArray with time not at first position: time is moved to front, spatial ordering preserved."""
        data = xr.DataArray(np.ones((4, 3, 6, 5), dtype=np.float64), dims=["z", "t", "x", "y"])
        field = TimeDependentField.from_xarray(data, "t", STANDARD_DIMS)
        assert isinstance(field, TimeDependentField)
        # time moved to front, spatial dims keep original order (z, x, y) → shape (3, 4, 6, 5)
        assert field.data.shape == (3, 4, 6, 5)
        assert field.axes == (ArrayAxis.Z, ArrayAxis.X, ArrayAxis.Y)

    def test_validation_error_missing_time_dim(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["t", "y", "x"])
        with pytest.raises(ValueError, match="Time dimension 'time' not found"):
            TimeDependentField.from_xarray(data, "time", {"y": ("Y", "center"), "x": ("X", "center")})

    def test_validation_error_missing_spatial_dim(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["t", "y", "x"])
        with pytest.raises(ValueError, match="Dimension 'x' in data is missing from dims mapping"):
            TimeDependentField.from_xarray(data, "t", {"y": ("Y", "center")})  # missing x

    def test_validation_error_extra_spatial_dim(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5)), dims=["t", "y", "x"])
        with pytest.raises(ValueError, match="Dimensions in dims mapping not found in data"):
            TimeDependentField.from_xarray(data, "t", STANDARD_DIMS)

    def test_validation_error_duplicate_axis(self) -> None:
        data = xr.DataArray(np.ones((3, 4, 5, 6)), dims=["t", "z", "y", "x"])
        with pytest.raises(ValueError, match="Axes must be unique"):
            TimeDependentField.from_xarray(
                data, "t", {"z": ("Z", "center"), "y": ("Z", "center"), "x": ("X", "center")}
            )  # both z and y map to Z axis

    def test_ignore_missing_dims_true_skips_extra_dims_mapping(self) -> None:
        """When ignore_missing_dims=True, dims entries absent from the data are silently ignored."""
        data = xr.DataArray(np.ones((3, 4, 5), dtype=np.float64), dims=["t", "y", "x"])
        field = TimeDependentField.from_xarray(
            data,
            "t",
            STANDARD_DIMS,
            ignore_missing_dims=True,
        )
        assert isinstance(field, TimeDependentField)
        assert field.spatial_shape == (4, 5)
        assert ArrayAxis.Z not in field.axes
