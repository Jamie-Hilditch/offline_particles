"""Tests that from_numpy and from_dask constructors check the type of the input array."""

import dask.array as da
import numpy as np
import pytest

from offline_particles.fields import StaticField, TimeDependentField
from offline_particles.spatial_arrays import ChunkedDaskArray, Stagger

STAGGER_ARGS = ("center", "center", "center")


class TestStaticFieldFromNumpy:
    def test_accepts_numpy_array(self) -> None:
        data = np.ones((4, 5, 6), dtype=np.float64)
        field = StaticField.from_numpy(data, *STAGGER_ARGS)
        assert isinstance(field, StaticField)

    def test_rejects_dask_array(self) -> None:
        data = da.ones((4, 5, 6), dtype=np.float64)
        with pytest.raises(TypeError, match="NumPy array"):
            StaticField.from_numpy(data, *STAGGER_ARGS)  # type: ignore[arg-type]

    def test_rejects_list(self) -> None:
        data = [[[1.0, 2.0], [3.0, 4.0]]]
        with pytest.raises(TypeError, match="NumPy array"):
            StaticField.from_numpy(data, *STAGGER_ARGS)  # type: ignore[arg-type]


class TestStaticFieldFromDask:
    def test_accepts_dask_array(self) -> None:
        data = da.ones((4, 5, 6), dtype=np.float64, chunks=(4, 5, 6))
        field = StaticField.from_dask(data, *STAGGER_ARGS)
        assert isinstance(field, StaticField)

    def test_rejects_numpy_array(self) -> None:
        data = np.ones((4, 5, 6), dtype=np.float64)
        with pytest.raises(TypeError, match="Dask array"):
            StaticField.from_dask(data, *STAGGER_ARGS)  # type: ignore[arg-type]

    def test_rejects_list(self) -> None:
        data = [[[1.0, 2.0], [3.0, 4.0]]]
        with pytest.raises(TypeError, match="Dask array"):
            StaticField.from_dask(data, *STAGGER_ARGS)  # type: ignore[arg-type]


class TestTimeDependentFieldFromNumpy:
    def test_accepts_numpy_array(self) -> None:
        data = np.ones((3, 4, 5, 6), dtype=np.float64)
        field = TimeDependentField.from_numpy(data, *STAGGER_ARGS)
        assert isinstance(field, TimeDependentField)

    def test_rejects_dask_array(self) -> None:
        data = da.ones((3, 4, 5, 6), dtype=np.float64)
        with pytest.raises(TypeError, match="NumPy array"):
            TimeDependentField.from_numpy(data, *STAGGER_ARGS)  # type: ignore[arg-type]

    def test_rejects_list(self) -> None:
        data = [[[[1.0, 2.0]], [[3.0, 4.0]]], [[[5.0, 6.0]], [[7.0, 8.0]]]]
        with pytest.raises(TypeError, match="NumPy array"):
            TimeDependentField.from_numpy(data, *STAGGER_ARGS)  # type: ignore[arg-type]


class TestTimeDependentFieldFromDask:
    def test_accepts_dask_array(self) -> None:
        data = da.ones((3, 4, 5, 6), dtype=np.float64, chunks=(1, 4, 5, 6))
        field = TimeDependentField.from_dask(data, *STAGGER_ARGS)
        assert isinstance(field, TimeDependentField)

    def test_rejects_numpy_array(self) -> None:
        data = np.ones((3, 4, 5, 6), dtype=np.float64)
        with pytest.raises(TypeError, match="Dask array"):
            TimeDependentField.from_dask(data, *STAGGER_ARGS)  # type: ignore[arg-type]

    def test_rejects_list(self) -> None:
        data = [[[[1.0, 2.0]], [[3.0, 4.0]]], [[[5.0, 6.0]], [[7.0, 8.0]]]]
        with pytest.raises(TypeError, match="Dask array"):
            TimeDependentField.from_dask(data, *STAGGER_ARGS)  # type: ignore[arg-type]


class TestChunkedDaskArrayTypeCheck:
    def test_accepts_dask_array(self) -> None:
        data = da.ones((4, 5, 6), dtype=np.float64, chunks=(4, 5, 6))
        arr = ChunkedDaskArray(data, Stagger.CENTER, Stagger.CENTER, Stagger.CENTER)
        assert arr.shape == (4, 5, 6)

    def test_rejects_numpy_array(self) -> None:
        data = np.ones((4, 5, 6), dtype=np.float64)
        with pytest.raises(TypeError, match="Dask array"):
            ChunkedDaskArray(data, Stagger.CENTER, Stagger.CENTER, Stagger.CENTER)  # type: ignore[arg-type]

    def test_rejects_list(self) -> None:
        data = [[[1.0, 2.0], [3.0, 4.0]]]
        with pytest.raises(TypeError, match="Dask array"):
            ChunkedDaskArray(data, Stagger.CENTER, Stagger.CENTER, Stagger.CENTER)  # type: ignore[arg-type]


class TestStaticFieldFromArraylike:
    def test_accepts_list(self) -> None:
        data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        field = StaticField.from_arraylike(data, *STAGGER_ARGS)
        assert isinstance(field, StaticField)

    def test_accepts_numpy_array(self) -> None:
        data = np.ones((4, 5, 6), dtype=np.float64)
        field = StaticField.from_arraylike(data, *STAGGER_ARGS)
        assert isinstance(field, StaticField)

    def test_converts_to_numpy(self) -> None:
        data = [[1.0, 2.0], [3.0, 4.0]]
        field = StaticField.from_arraylike(data, "invariant", "center", "center")
        assert isinstance(field, StaticField)
        assert field.dtype == np.float64


class TestTimeDependentFieldFromArraylike:
    def test_accepts_list(self) -> None:
        data = [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        ]
        field = TimeDependentField.from_arraylike(data, *STAGGER_ARGS)
        assert isinstance(field, TimeDependentField)

    def test_accepts_numpy_array(self) -> None:
        data = np.ones((3, 4, 5, 6), dtype=np.float64)
        field = TimeDependentField.from_arraylike(data, *STAGGER_ARGS)
        assert isinstance(field, TimeDependentField)

    def test_converts_to_numpy(self) -> None:
        data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        field = TimeDependentField.from_arraylike(data, "invariant", "center", "center")
        assert isinstance(field, TimeDependentField)
        assert field.dtype == np.float64
