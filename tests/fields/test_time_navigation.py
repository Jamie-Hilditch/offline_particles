"""Tests for TimeDependentField time-index navigation.

Increment/decrement/set_time_index and the preload_space option on from_dask.
"""

import dask.array as da
import numpy as np
import pytest

from offline_particles.fields import TimeDependentField
from offline_particles.spatial_arrays import BBox, ChunkedDaskArray, NumpyArray


def _full_domain_bbox(nx: int) -> BBox:
    return BBox(zmin=0.0, zmax=0.0, ymin=0.0, ymax=0.0, xmin=0.0, xmax=float(nx - 1))


def _make_field(num_timesteps: int, nx: int = 3) -> TimeDependentField:
    # data[t, :] == t, so the loaded slices can be checked by value.
    data = np.arange(num_timesteps, dtype=np.float64)[:, None] * np.ones((1, nx))
    return TimeDependentField.from_numpy(data, ("X",), ("center",))


class TestIncrementTime:
    def test_advances_previous_and_next_slices(self) -> None:
        field = _make_field(4)

        field.increment_time()

        assert field._It == 1
        previous_data, _ = field.previous_time_slice.get_data_subset(_full_domain_bbox(nx=3))
        next_data, _ = field.next_time_slice.get_data_subset(_full_domain_bbox(nx=3))
        np.testing.assert_array_equal(previous_data, np.full(3, 1.0))
        np.testing.assert_array_equal(next_data, np.full(3, 2.0))

    def test_invalidates_delta_and_output_caches(self) -> None:
        field = _make_field(4)
        field.get_field_data(0.5, _full_domain_bbox(nx=3))
        assert field._output_valid is True
        assert field._cached_delta_valid is True

        field.increment_time()

        assert field._output_valid is False
        assert field._cached_delta_valid is False

    def test_raises_at_penultimate_timestep(self) -> None:
        field = _make_field(3)  # valid time indices are 0 and 1
        field.increment_time()  # now at It == 1, the penultimate timestep

        with pytest.raises(IndexError, match="Cannot increment past the penultimate timestep"):
            field.increment_time()


class TestDecrementTime:
    def test_moves_previous_and_next_slices_back(self) -> None:
        field = _make_field(4)
        field.increment_time()  # It == 1

        field.decrement_time()

        assert field._It == 0
        previous_data, _ = field.previous_time_slice.get_data_subset(_full_domain_bbox(nx=3))
        next_data, _ = field.next_time_slice.get_data_subset(_full_domain_bbox(nx=3))
        np.testing.assert_array_equal(previous_data, np.full(3, 0.0))
        np.testing.assert_array_equal(next_data, np.full(3, 1.0))

    def test_invalidates_delta_and_output_caches(self) -> None:
        field = _make_field(4)
        field.increment_time()  # It == 1
        field.get_field_data(1.5, _full_domain_bbox(nx=3))  # stays at It == 1
        assert field._output_valid is True

        field.decrement_time()

        assert field._output_valid is False
        assert field._cached_delta_valid is False

    def test_raises_at_first_timestep(self) -> None:
        field = _make_field(4)  # starts at It == 0

        with pytest.raises(IndexError, match="Cannot decrement past the first timestep"):
            field.decrement_time()


class TestSetTimeIndex:
    def test_same_index_is_noop(self) -> None:
        field = _make_field(4)
        field.get_field_data(0.5, _full_domain_bbox(nx=3))
        output_before = field._output

        field.set_time_index(0)

        assert field._It == 0
        assert field._output is output_before
        assert field._output_valid is True

    def test_next_index_increments(self) -> None:
        field = _make_field(4)

        field.set_time_index(1)

        assert field._It == 1

    def test_previous_index_decrements(self) -> None:
        field = _make_field(4)
        field.set_time_index(1)

        field.set_time_index(0)

        assert field._It == 0

    def test_direct_jump_loads_correct_slices(self) -> None:
        field = _make_field(5)  # valid time indices are 0..3

        field.set_time_index(3)

        assert field._It == 3
        previous_data, _ = field.previous_time_slice.get_data_subset(_full_domain_bbox(nx=3))
        next_data, _ = field.next_time_slice.get_data_subset(_full_domain_bbox(nx=3))
        np.testing.assert_array_equal(previous_data, np.full(3, 3.0))
        np.testing.assert_array_equal(next_data, np.full(3, 4.0))

    def test_raises_for_negative_index(self) -> None:
        field = _make_field(5)  # valid range is 0..3
        field.set_time_index(2)  # move away from 0 so -1 isn't treated as a simple decrement
        with pytest.raises(IndexError, match=r"Valid range of time indices is 0,\.\.\.,3"):
            field.set_time_index(-1)

    def test_raises_for_index_too_large(self) -> None:
        field = _make_field(5)  # valid range is 0..3
        with pytest.raises(IndexError, match=r"Valid range of time indices is 0,\.\.\.,3"):
            field.set_time_index(4)


class TestFromDaskPreloadSpace:
    def test_defaults_to_chunked_dask_array(self) -> None:
        data = da.ones((3, 4), dtype=np.float64, chunks=(1, 4))
        field = TimeDependentField.from_dask(data, ("X",), ("center",))
        assert isinstance(field.previous_time_slice, ChunkedDaskArray)

    def test_preload_space_true_uses_numpy_array(self) -> None:
        data = da.ones((3, 4), dtype=np.float64, chunks=(1, 4))
        field = TimeDependentField.from_dask(data, ("X",), ("center",), preload_space=True)
        assert isinstance(field.previous_time_slice, NumpyArray)
