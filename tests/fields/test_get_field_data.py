"""Tests for Field.get_field_data on static and time-dependent fields."""

import dask.array as da
import numpy as np

from offline_particles.fields import StaticField, TimeDependentField
from offline_particles.spatial_arrays import ArrayLayout, BBox, NumpyArray


def _full_domain_bbox(nz: int = 1, ny: int = 1, nx: int = 1) -> BBox:
    return BBox(zmin=0.0, zmax=float(nz - 1), ymin=0.0, ymax=float(ny - 1), xmin=0.0, xmax=float(nx - 1))


class TestStaticFieldGetFieldData:
    def test_numpy_backed_field_returns_full_data_and_offsets(self) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        field = StaticField.from_numpy(data, axes=("X",), staggers=("center",))

        field_data = field.get_field_data(12.5, _full_domain_bbox(nx=3))

        np.testing.assert_array_equal(field_data.array, data)
        assert field_data.offsets == (0.0,)

    def test_dask_backed_field_returns_computed_data_and_offsets(self) -> None:
        data = da.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), chunks=(1, 2))
        field = StaticField.from_dask(data, axes=("Y", "X"), staggers=("center", "center"))

        field_data = field.get_field_data(-3.0, _full_domain_bbox(ny=2, nx=2))

        np.testing.assert_array_equal(field_data.array, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        assert field_data.offsets == (0.0, 0.0)


class TestTimeDependentFieldGetFieldData:
    def test_numpy_backed_field_interpolates_between_timesteps(self) -> None:
        data = np.array(
            [
                [0.0, 10.0, 20.0],
                [10.0, 20.0, 30.0],
            ],
            dtype=np.float32,
        )
        field = TimeDependentField.from_numpy(data, axes=("X",), staggers=("center",))

        field_data = field.get_field_data(0.5, _full_domain_bbox(nx=3))

        np.testing.assert_array_equal(field_data.array, np.array([5.0, 15.0, 25.0], dtype=np.float32))
        assert field_data.offsets == (0.0,)

    def test_dask_backed_field_interpolates_between_timesteps(self) -> None:
        data = da.from_array(
            np.array(
                [
                    [0.0, 10.0, 20.0],
                    [10.0, 20.0, 30.0],
                ],
                dtype=np.float32,
            ),
            chunks=(1, 3),
        )
        field = TimeDependentField.from_dask(data, axes=("X",), staggers=("center",))

        field_data = field.get_field_data(0.5, _full_domain_bbox(nx=3))

        np.testing.assert_array_equal(field_data.array, np.array([5.0, 15.0, 25.0], dtype=np.float32))
        assert field_data.offsets == (0.0,)

    def test_reuses_output_cache_for_repeated_calls_with_same_time_and_bbox(self) -> None:
        data = da.from_array(
            np.array(
                [
                    [0.0, 10.0, 20.0, 30.0],
                    [10.0, 20.0, 30.0, 40.0],
                ],
                dtype=np.float32,
            ),
            chunks=(1, 2),
        )
        field = TimeDependentField.from_dask(data, axes=("X",), staggers=("center",))
        bbox = _full_domain_bbox(nx=4)

        first_field_data = field.get_field_data(0.5, bbox)
        first_output = field._output

        second_field_data = field.get_field_data(0.5, bbox)

        np.testing.assert_array_equal(first_field_data.array, np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32))
        assert second_field_data.array is first_output
        assert second_field_data.array is first_field_data.array
        assert field._output_valid is True
        assert field._cached_delta_valid is True
        assert field._cached_offsets == (0.0,)
        assert field._array_shape == (4,)

    def test_keeps_delta_cache_when_only_fractional_time_changes(self) -> None:
        data = da.from_array(
            np.array(
                [
                    [0.0, 10.0, 20.0],
                    [10.0, 20.0, 30.0],
                ],
                dtype=np.float32,
            ),
            chunks=(1, 3),
        )
        field = TimeDependentField.from_dask(data, axes=("X",), staggers=("center",))
        bbox = _full_domain_bbox(nx=3)

        first_field_data = field.get_field_data(0.25, bbox)
        first_values = first_field_data.array.copy()
        first_delta = field._delta

        second_field_data = field.get_field_data(0.75, bbox)

        np.testing.assert_array_equal(first_values, np.array([2.5, 12.5, 22.5], dtype=np.float32))
        np.testing.assert_array_equal(second_field_data.array, np.array([7.5, 17.5, 27.5], dtype=np.float32))
        assert field._cached_delta_valid is True
        assert field._output_valid is True
        assert field._delta is first_delta
        assert field._cached_offsets == (0.0,)

    def test_reallocates_interpolation_arrays_when_bbox_size_changes(self) -> None:
        data = da.from_array(
            np.array(
                [
                    [0.0, 10.0, 20.0, 30.0],
                    [10.0, 20.0, 30.0, 40.0],
                ],
                dtype=np.float32,
            ),
            chunks=(1, 2),
        )
        field = TimeDependentField.from_dask(data, axes=("X",), staggers=("center",))

        smaller_bbox = _full_domain_bbox(nx=2)
        larger_bbox = _full_domain_bbox(nx=4)

        smaller_field_data = field.get_field_data(0.5, smaller_bbox)
        smaller_delta = field._delta
        smaller_output = field._output
        larger_field_data = field.get_field_data(0.5, larger_bbox)

        np.testing.assert_array_equal(smaller_field_data.array, np.array([5.0, 15.0], dtype=np.float32))
        np.testing.assert_array_equal(larger_field_data.array, np.array([5.0, 15.0, 25.0, 35.0], dtype=np.float32))
        assert smaller_field_data.offsets == (0.0,)
        assert larger_field_data.offsets == (0.0,)
        assert field._array_shape == (4,)
        assert field._cached_delta_valid is True
        assert field._output_valid is True
        assert field._delta is not smaller_delta
        assert field._output is not smaller_output

    def test_output_dtype_is_respected_during_interpolation(self) -> None:
        data = np.array(
            [
                [0.0, 1.0],
                [2.0, 3.0],
            ],
            dtype=np.float32,
        )
        field = TimeDependentField(data, ArrayLayout(("X",), ("center",)), NumpyArray, output_dtype=np.float64)

        field_data = field.get_field_data(0.25, _full_domain_bbox(nx=2))

        assert field_data.array.dtype == np.float64
        np.testing.assert_allclose(field_data.array, np.array([0.5, 1.5], dtype=np.float64))
        assert field_data.offsets == (0.0,)
