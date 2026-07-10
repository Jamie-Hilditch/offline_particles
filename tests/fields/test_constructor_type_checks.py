"""Tests that from_numpy and from_dask constructors check the type of the input array."""

import dask.array as da
import numpy as np
import pytest

from offline_particles.fields import StaticField, TimeDependentField
from offline_particles.spatial_arrays import ArrayLayout, ChunkedDaskArray

AXES_ARGS = ("Z", "Y", "X")
STAGGER_ARGS = ("center", "center", "center")
LAYOUT = ArrayLayout(AXES_ARGS, STAGGER_ARGS)

AXES_ARGS_2D = ("Y", "X")
STAGGER_ARGS_2D = ("center", "center")
LAYOUT_2D = ArrayLayout(AXES_ARGS_2D, STAGGER_ARGS_2D)


class TestStaticFieldFromNumpy:
    def test_accepts_numpy_array(self) -> None:
        data = np.ones((4, 5, 6), dtype=np.float64)
        field = StaticField.from_numpy(data, AXES_ARGS, STAGGER_ARGS)
        assert isinstance(field, StaticField)

    def test_rejects_dask_array(self) -> None:
        data = da.ones((4, 5, 6), dtype=np.float64)
        with pytest.raises(TypeError, match="NumPy array"):
            StaticField.from_numpy(data, AXES_ARGS, STAGGER_ARGS)  # type: ignore[arg-type]

    def test_rejects_list(self) -> None:
        data = [[[1.0, 2.0], [3.0, 4.0]]]
        with pytest.raises(TypeError, match="NumPy array"):
            StaticField.from_numpy(data, AXES_ARGS, STAGGER_ARGS)  # type: ignore[arg-type]


class TestStaticFieldFromDask:
    def test_accepts_dask_array(self) -> None:
        data = da.ones((4, 5, 6), dtype=np.float64, chunks=(4, 5, 6))
        field = StaticField.from_dask(data, AXES_ARGS, STAGGER_ARGS)
        assert isinstance(field, StaticField)

    def test_rejects_numpy_array(self) -> None:
        data = np.ones((4, 5, 6), dtype=np.float64)
        with pytest.raises(TypeError, match="Dask array"):
            StaticField.from_dask(data, AXES_ARGS, STAGGER_ARGS)  # type: ignore[arg-type]

    def test_rejects_list(self) -> None:
        data = [[[1.0, 2.0], [3.0, 4.0]]]
        with pytest.raises(TypeError, match="Dask array"):
            StaticField.from_dask(data, AXES_ARGS, STAGGER_ARGS)  # type: ignore[arg-type]


class TestTimeDependentFieldFromNumpy:
    def test_accepts_numpy_array(self) -> None:
        data = np.ones((3, 4, 5, 6), dtype=np.float64)
        field = TimeDependentField.from_numpy(data, AXES_ARGS, STAGGER_ARGS)
        assert isinstance(field, TimeDependentField)

    def test_rejects_dask_array(self) -> None:
        data = da.ones((3, 4, 5, 6), dtype=np.float64)
        with pytest.raises(TypeError, match="NumPy array"):
            TimeDependentField.from_numpy(data, AXES_ARGS, STAGGER_ARGS)  # type: ignore[arg-type]

    def test_rejects_list(self) -> None:
        data = [[[[1.0, 2.0]], [[3.0, 4.0]]], [[[5.0, 6.0]], [[7.0, 8.0]]]]
        with pytest.raises(TypeError, match="NumPy array"):
            TimeDependentField.from_numpy(data, AXES_ARGS, STAGGER_ARGS)  # type: ignore[arg-type]


class TestTimeDependentFieldFromDask:
    def test_accepts_dask_array(self) -> None:
        data = da.ones((3, 4, 5, 6), dtype=np.float64, chunks=(1, 4, 5, 6))
        field = TimeDependentField.from_dask(data, AXES_ARGS, STAGGER_ARGS)
        assert isinstance(field, TimeDependentField)

    def test_rejects_numpy_array(self) -> None:
        data = np.ones((3, 4, 5, 6), dtype=np.float64)
        with pytest.raises(TypeError, match="Dask array"):
            TimeDependentField.from_dask(data, AXES_ARGS, STAGGER_ARGS)  # type: ignore[arg-type]

    def test_rejects_list(self) -> None:
        data = [[[[1.0, 2.0]], [[3.0, 4.0]]], [[[5.0, 6.0]], [[7.0, 8.0]]]]
        with pytest.raises(TypeError, match="Dask array"):
            TimeDependentField.from_dask(data, AXES_ARGS, STAGGER_ARGS)  # type: ignore[arg-type]


class TestChunkedDaskArrayTypeCheck:
    def test_accepts_dask_array(self) -> None:
        data = da.ones((4, 5, 6), dtype=np.float64, chunks=(4, 5, 6))
        arr = ChunkedDaskArray(data, LAYOUT)
        assert arr.shape == (4, 5, 6)

    def test_rejects_numpy_array(self) -> None:
        data = np.ones((4, 5, 6), dtype=np.float64)
        with pytest.raises(TypeError, match="Dask array"):
            ChunkedDaskArray(data, LAYOUT)  # type: ignore[arg-type]

    def test_rejects_list(self) -> None:
        data = [[[1.0, 2.0], [3.0, 4.0]]]
        with pytest.raises(TypeError, match="Dask array"):
            ChunkedDaskArray(data, LAYOUT)  # type: ignore[arg-type]


class TestStaticFieldFromArraylike:
    def test_accepts_list(self) -> None:
        data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        field = StaticField.from_arraylike(data, AXES_ARGS, STAGGER_ARGS)
        assert isinstance(field, StaticField)

    def test_accepts_numpy_array(self) -> None:
        data = np.ones((4, 5, 6), dtype=np.float64)
        field = StaticField.from_arraylike(data, AXES_ARGS, STAGGER_ARGS)
        assert isinstance(field, StaticField)

    def test_converts_to_numpy(self) -> None:
        data = [[1.0, 2.0], [3.0, 4.0]]
        field = StaticField.from_arraylike(data, AXES_ARGS_2D, STAGGER_ARGS_2D)
        assert isinstance(field, StaticField)
        assert field.dtype == np.float64

    def test_warns_on_dask_array(self) -> None:
        data = da.ones((4, 5, 6), dtype=np.float64, chunks=(4, 5, 6))
        with pytest.warns(UserWarning, match="dask.array.Array"):
            field = StaticField.from_arraylike(data, AXES_ARGS, STAGGER_ARGS)
        assert isinstance(field, StaticField)


class TestTimeDependentFieldFromArraylike:
    def test_accepts_list(self) -> None:
        data = [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        ]
        field = TimeDependentField.from_arraylike(data, AXES_ARGS, STAGGER_ARGS)
        assert isinstance(field, TimeDependentField)

    def test_accepts_numpy_array(self) -> None:
        data = np.ones((3, 4, 5, 6), dtype=np.float64)
        field = TimeDependentField.from_arraylike(data, AXES_ARGS, STAGGER_ARGS)
        assert isinstance(field, TimeDependentField)

    def test_converts_to_numpy(self) -> None:
        data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        field = TimeDependentField.from_arraylike(data, AXES_ARGS_2D, STAGGER_ARGS_2D)
        assert isinstance(field, TimeDependentField)
        assert field.dtype == np.float64

    def test_warns_on_dask_array(self) -> None:
        data = da.ones((3, 4, 5, 6), dtype=np.float64, chunks=(1, 4, 5, 6))
        with pytest.warns(UserWarning, match="dask.array.Array"):
            field = TimeDependentField.from_arraylike(data, AXES_ARGS, STAGGER_ARGS)
        assert isinstance(field, TimeDependentField)


class TestZeroSpatialDimValidation:
    """Fields must have at least one spatial dimension."""

    def test_static_field_rejects_zero_spatial_dims_from_numpy(self) -> None:
        """StaticField.from_numpy raises ValueError when no axes are provided."""
        data = np.array(1.0)  # 0-d ndarray
        with pytest.raises(ValueError, match="at least 1 spatial dimension"):
            StaticField.from_numpy(data, (), ())  # type: ignore[arg-type]

    def test_static_field_rejects_zero_spatial_dims_from_arraylike(self) -> None:
        """StaticField.from_arraylike raises ValueError when no axes are provided."""
        with pytest.raises(ValueError, match="at least 1 spatial dimension"):
            StaticField.from_arraylike(1.0, (), ())  # type: ignore[arg-type]

    def test_static_field_rejects_zero_spatial_dims_from_dask(self) -> None:
        """StaticField.from_dask raises ValueError when no axes are provided."""
        data = da.from_array(np.array(1.0))  # 0-d dask array
        with pytest.raises(ValueError, match="at least 1 spatial dimension"):
            StaticField.from_dask(data, (), ())  # type: ignore[arg-type]

    def test_time_dependent_field_rejects_zero_spatial_dims(self) -> None:
        """TimeDependentField raises ValueError for a time-only (1-D) array with no spatial dims."""
        data = np.ones((3,), dtype=np.float64)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            TimeDependentField.from_numpy(data, (), ())


class TestMinimumTimestepsValidation:
    """TimeDependentField requires at least 2 timesteps to interpolate between."""

    def test_rejects_single_timestep_from_numpy(self) -> None:
        data = np.ones((1, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="at least 2 time steps"):
            TimeDependentField.from_numpy(data, ("X",), ("center",))

    def test_rejects_single_timestep_from_dask(self) -> None:
        data = da.ones((1, 3), dtype=np.float64, chunks=(1, 3))
        with pytest.raises(ValueError, match="at least 2 time steps"):
            TimeDependentField.from_dask(data, ("X",), ("center",))
