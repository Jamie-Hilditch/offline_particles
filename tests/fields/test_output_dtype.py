"""Tests that output_dtype is threaded through the TimeDependentField alternative constructors."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from offline_particles.fields import TimeDependentField

AXES_ARGS = ("X",)
STAGGER_ARGS = ("center",)


class TestFromNumpyOutputDtype:
    def test_defaults_to_data_dtype(self) -> None:
        data = np.ones((2, 3), dtype=np.float32)
        field = TimeDependentField.from_numpy(data, AXES_ARGS, STAGGER_ARGS)
        assert field.output_dtype == np.float32

    def test_explicit_output_dtype_is_respected(self) -> None:
        data = np.ones((2, 3), dtype=np.float32)
        field = TimeDependentField.from_numpy(data, AXES_ARGS, STAGGER_ARGS, output_dtype=np.float64)
        assert field.output_dtype == np.float64
        assert field.dtype == np.float32


class TestFromDaskOutputDtype:
    def test_defaults_to_data_dtype(self) -> None:
        data = da.ones((2, 3), dtype=np.float32, chunks=(1, 3))
        field = TimeDependentField.from_dask(data, AXES_ARGS, STAGGER_ARGS)
        assert field.output_dtype == np.float32

    def test_explicit_output_dtype_is_respected(self) -> None:
        data = da.ones((2, 3), dtype=np.float32, chunks=(1, 3))
        field = TimeDependentField.from_dask(data, AXES_ARGS, STAGGER_ARGS, output_dtype=np.float64)
        assert field.output_dtype == np.float64
        assert field.dtype == np.float32


class TestFromArraylikeOutputDtype:
    def test_defaults_to_data_dtype(self) -> None:
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        field = TimeDependentField.from_arraylike(data, AXES_ARGS, STAGGER_ARGS)
        assert field.output_dtype == field.dtype

    def test_explicit_output_dtype_is_respected(self) -> None:
        data = np.ones((2, 3), dtype=np.float32)
        field = TimeDependentField.from_arraylike(data, AXES_ARGS, STAGGER_ARGS, output_dtype=np.float64)
        assert field.output_dtype == np.float64
        assert field.dtype == np.float32


class TestFromXarrayOutputDtype:
    def test_numpy_backed_defaults_to_data_dtype(self) -> None:
        data = xr.DataArray(np.ones((2, 3), dtype=np.float32), dims=["t", "x"])
        field = TimeDependentField.from_xarray(data, "t", {"x": ("X", "center")})
        assert field.output_dtype == np.float32

    def test_numpy_backed_explicit_output_dtype_is_respected(self) -> None:
        data = xr.DataArray(np.ones((2, 3), dtype=np.float32), dims=["t", "x"])
        field = TimeDependentField.from_xarray(data, "t", {"x": ("X", "center")}, output_dtype=np.float64)
        assert field.output_dtype == np.float64
        assert field.dtype == np.float32

    def test_dask_backed_defaults_to_data_dtype(self) -> None:
        data = xr.DataArray(da.ones((2, 3), dtype=np.float32, chunks=(1, 3)), dims=["t", "x"])
        field = TimeDependentField.from_xarray(data, "t", {"x": ("X", "center")})
        assert field.output_dtype == np.float32

    def test_dask_backed_explicit_output_dtype_is_respected(self) -> None:
        data = xr.DataArray(da.ones((2, 3), dtype=np.float32, chunks=(1, 3)), dims=["t", "x"])
        field = TimeDependentField.from_xarray(data, "t", {"x": ("X", "center")}, output_dtype=np.float64)
        assert field.output_dtype == np.float64
        assert field.dtype == np.float32


class TestOutputDtypeAffectsInterpolatedOutput:
    """End-to-end check that a constructor-supplied output_dtype is used when computing field data."""

    def test_from_numpy_output_dtype_used_in_get_field_data(self, full_domain_bbox) -> None:
        data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        field = TimeDependentField.from_numpy(data, AXES_ARGS, STAGGER_ARGS, output_dtype=np.float64)

        field_data = field.get_field_data(0.25, full_domain_bbox(nx=2))

        assert field_data.array.dtype == np.float64
        np.testing.assert_allclose(field_data.array, np.array([0.5, 1.5], dtype=np.float64))

    def test_from_xarray_output_dtype_used_in_get_field_data(self, full_domain_bbox) -> None:
        data = xr.DataArray(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32), dims=["t", "x"])
        field = TimeDependentField.from_xarray(data, "t", {"x": ("X", "center")}, output_dtype=np.float64)

        field_data = field.get_field_data(0.25, full_domain_bbox(nx=2))

        assert field_data.array.dtype == np.float64
        np.testing.assert_allclose(field_data.array, np.array([0.5, 1.5], dtype=np.float64))


class TestInvalidOutputDtypeRaises:
    @pytest.mark.parametrize("constructor_name", ["from_numpy", "from_dask", "from_arraylike"])
    def test_invalid_output_dtype_raises_type_error(self, constructor_name: str) -> None:
        data: np.ndarray | da.Array
        if constructor_name == "from_dask":
            data = da.ones((2, 3), dtype=np.float32, chunks=(1, 3))
        else:
            data = np.ones((2, 3), dtype=np.float32)
        constructor = getattr(TimeDependentField, constructor_name)
        with pytest.raises(TypeError):
            constructor(data, AXES_ARGS, STAGGER_ARGS, output_dtype="not-a-dtype")
