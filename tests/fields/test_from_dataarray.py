"""Tests for the field_from_dataarray helper function."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from offline_particles.fields import StaticField, TimeDependentField, field_from_dataarray
from offline_particles.spatial_arrays import Dimension, Stagger

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DIM_MAP_3D = {
    "z": Dimension.Z_CENTER,
    "y": Dimension.Y_CENTER,
    "x": Dimension.X_CENTER,
}

DIM_MAP_4D = {
    "time": Dimension.TIME,
    **DIM_MAP_3D,
}


def _numpy_static(shape: tuple[int, ...], dims: list[str]) -> xr.DataArray:
    return xr.DataArray(np.ones(shape, dtype=np.float64), dims=dims)


def _numpy_time_dep(shape: tuple[int, ...], dims: list[str]) -> xr.DataArray:
    return xr.DataArray(np.ones(shape, dtype=np.float64), dims=dims)


def _dask_static(shape: tuple[int, ...], dims: list[str]) -> xr.DataArray:
    return xr.DataArray(da.ones(shape, dtype=np.float64, chunks=shape), dims=dims)


def _dask_time_dep(shape: tuple[int, ...], dims: list[str]) -> xr.DataArray:
    chunks = (1,) + shape[1:]
    return xr.DataArray(da.ones(shape, dtype=np.float64, chunks=chunks), dims=dims)


# ---------------------------------------------------------------------------
# Return type detection
# ---------------------------------------------------------------------------


class TestFieldFromDataarrayReturnType:
    def test_no_time_dim_returns_static_field(self) -> None:
        da_arr = _numpy_static((5, 6, 7), ["z", "y", "x"])
        field = field_from_dataarray(da_arr, DIM_MAP_3D)
        assert isinstance(field, StaticField)

    def test_with_time_dim_returns_time_dep_field(self) -> None:
        da_arr = _numpy_time_dep((3, 5, 6, 7), ["time", "z", "y", "x"])
        field = field_from_dataarray(da_arr, DIM_MAP_4D)
        assert isinstance(field, TimeDependentField)

    def test_dask_no_time_returns_static_field(self) -> None:
        da_arr = _dask_static((5, 6, 7), ["z", "y", "x"])
        field = field_from_dataarray(da_arr, DIM_MAP_3D)
        assert isinstance(field, StaticField)

    def test_dask_with_time_returns_time_dep_field(self) -> None:
        da_arr = _dask_time_dep((3, 5, 6, 7), ["time", "z", "y", "x"])
        field = field_from_dataarray(da_arr, DIM_MAP_4D)
        assert isinstance(field, TimeDependentField)


# ---------------------------------------------------------------------------
# Stagger assignment
# ---------------------------------------------------------------------------


class TestFieldFromDataarrayStagger:
    def test_center_stagger_on_all_dims(self) -> None:
        da_arr = _numpy_static((5, 6, 7), ["z", "y", "x"])
        field = field_from_dataarray(da_arr, DIM_MAP_3D)
        assert field.z_stagger is Stagger.CENTER
        assert field.y_stagger is Stagger.CENTER
        assert field.x_stagger is Stagger.CENTER

    def test_inner_x_stagger(self) -> None:
        da_arr = _numpy_static((5, 6, 6), ["z", "y", "x_inner"])
        dim_map = {
            "z": Dimension.Z_CENTER,
            "y": Dimension.Y_CENTER,
            "x_inner": Dimension.X_INNER,
        }
        field = field_from_dataarray(da_arr, dim_map)
        assert field.x_stagger is Stagger.INNER

    def test_outer_z_stagger(self) -> None:
        da_arr = _numpy_static((6, 6, 7), ["z_outer", "y", "x"])
        dim_map = {
            "z_outer": Dimension.Z_OUTER,
            "y": Dimension.Y_CENTER,
            "x": Dimension.X_CENTER,
        }
        field = field_from_dataarray(da_arr, dim_map)
        assert field.z_stagger is Stagger.OUTER

    def test_missing_z_dim_gives_invariant_z_stagger(self) -> None:
        da_arr = _numpy_static((6, 7), ["y", "x"])
        field = field_from_dataarray(da_arr, DIM_MAP_3D)
        assert field.z_stagger is Stagger.INVARIANT
        assert field.y_stagger is Stagger.CENTER
        assert field.x_stagger is Stagger.CENTER

    def test_missing_z_and_y_gives_invariant(self) -> None:
        da_arr = _numpy_static((7,), ["x"])
        field = field_from_dataarray(da_arr, DIM_MAP_3D)
        assert field.z_stagger is Stagger.INVARIANT
        assert field.y_stagger is Stagger.INVARIANT
        assert field.x_stagger is Stagger.CENTER

    def test_alias_dimension_gives_correct_stagger(self) -> None:
        da_arr = _numpy_static((5, 6, 7), ["depth", "eta", "xi"])
        dim_map = {
            "depth": Dimension.DEPTH_CENTER,
            "eta": Dimension.ETA_CENTER,
            "xi": Dimension.XI_CENTER,
        }
        field = field_from_dataarray(da_arr, dim_map)
        assert field.z_stagger is Stagger.CENTER
        assert field.y_stagger is Stagger.CENTER
        assert field.x_stagger is Stagger.CENTER


# ---------------------------------------------------------------------------
# Dimension reordering
# ---------------------------------------------------------------------------


class TestFieldFromDataarrayReordering:
    def test_out_of_order_dims_are_reordered(self) -> None:
        # DataArray with dims in (x, y, z) order – should be transposed to (z, y, x)
        da_arr = xr.DataArray(
            np.arange(5 * 6 * 7, dtype=np.float64).reshape(7, 6, 5),
            dims=["x", "y", "z"],
        )
        field = field_from_dataarray(da_arr, DIM_MAP_3D)
        # After reordering, spatial shape should be (5, 6, 7)
        assert field.spatial_shape == (5, 6, 7)

    def test_time_dim_not_first_is_reordered(self) -> None:
        # DataArray with time last
        da_arr = xr.DataArray(
            np.ones((5, 6, 7, 3), dtype=np.float64),
            dims=["z", "y", "x", "time"],
        )
        field = field_from_dataarray(da_arr, DIM_MAP_4D)
        assert isinstance(field, TimeDependentField)
        # TimeDependentField.spatial_shape excludes time
        assert field.spatial_shape == (5, 6, 7)


# ---------------------------------------------------------------------------
# Attribute handling
# ---------------------------------------------------------------------------


class TestFieldFromDataarrayAttrs:
    def test_attrs_inherited_from_dataarray(self) -> None:
        da_arr = xr.DataArray(
            np.ones((5, 6, 7)),
            dims=["z", "y", "x"],
            attrs={"units": "m/s", "long_name": "eastward velocity"},
        )
        field = field_from_dataarray(da_arr, DIM_MAP_3D)
        assert field.attrs["units"] == "m/s"
        assert field.attrs["long_name"] == "eastward velocity"

    def test_explicit_attrs_override_dataarray_attrs(self) -> None:
        da_arr = xr.DataArray(
            np.ones((5, 6, 7)),
            dims=["z", "y", "x"],
            attrs={"units": "m/s"},
        )
        field = field_from_dataarray(da_arr, DIM_MAP_3D, attrs={"units": "cm/s"})
        assert field.attrs["units"] == "cm/s"

    def test_empty_attrs_dict_is_accepted(self) -> None:
        da_arr = xr.DataArray(np.ones((5, 6, 7)), dims=["z", "y", "x"])
        field = field_from_dataarray(da_arr, DIM_MAP_3D, attrs={})
        assert field.attrs == {}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestFieldFromDataarrayErrors:
    def test_raises_type_error_for_non_dataarray(self) -> None:
        with pytest.raises(TypeError, match="DataArray"):
            field_from_dataarray(np.ones((5, 6, 7)), DIM_MAP_3D)  # type: ignore[arg-type]

    def test_raises_value_error_for_unmapped_dim(self) -> None:
        da_arr = xr.DataArray(np.ones((5, 6)), dims=["unmapped", "y"])
        with pytest.raises(ValueError, match="unmapped"):
            field_from_dataarray(da_arr, {"y": Dimension.Y_CENTER})

    def test_raises_value_error_for_duplicate_direction(self) -> None:
        da_arr = xr.DataArray(np.ones((5, 6)), dims=["y1", "y2"])
        dim_map = {
            "y1": Dimension.Y_CENTER,
            "y2": Dimension.Y_LEFT,
        }
        with pytest.raises(ValueError, match="direction 'Y'"):
            field_from_dataarray(da_arr, dim_map)
