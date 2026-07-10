"""Tests for Field.validate_shape and SimulationSize.axis_size."""

import numpy as np
import pytest

from offline_particles.fields import SimulationSize, StaticField, TimeDependentField
from offline_particles.spatial_arrays import ArrayAxis


class TestStaticFieldValidateShape:
    def test_passes_when_shape_matches(self) -> None:
        data = np.ones((4, 5, 6), dtype=np.float64)
        field = StaticField.from_numpy(data, ("Z", "Y", "X"), ("center", "center", "center"))
        field.validate_shape(SimulationSize(time=10, z=4, y=5, x=6))

    def test_raises_on_spatial_mismatch(self) -> None:
        data = np.ones((4, 5, 6), dtype=np.float64)
        field = StaticField.from_numpy(data, ("Z", "Y", "X"), ("center", "center", "center"))
        with pytest.raises(ValueError, match="Expected size 4 along axis Y but got 5"):
            field.validate_shape(SimulationSize(time=10, z=4, y=4, x=6))

    def test_accounts_for_stagger_when_computing_expected_size(self) -> None:
        # x axis has an outer stagger, so its expected size is N + 1 where N is the centered size.
        data = np.ones((4, 5, 7), dtype=np.float64)
        field = StaticField.from_numpy(data, ("Z", "Y", "X"), ("center", "center", "outer"))
        field.validate_shape(SimulationSize(time=10, z=4, y=5, x=6))

    def test_raises_when_stagger_size_mismatch(self) -> None:
        # x axis has an outer stagger, so it should have size N + 1 = 7, not 6.
        data = np.ones((4, 5, 6), dtype=np.float64)
        field = StaticField.from_numpy(data, ("Z", "Y", "X"), ("center", "center", "outer"))
        with pytest.raises(ValueError, match="Expected size 7 along axis X but got 6"):
            field.validate_shape(SimulationSize(time=10, z=4, y=5, x=6))


class TestTimeDependentFieldValidateShape:
    def test_passes_when_shape_matches(self) -> None:
        data = np.ones((3, 4, 5), dtype=np.float64)
        field = TimeDependentField.from_numpy(data, ("Y", "X"), ("center", "center"))
        field.validate_shape(SimulationSize(time=3, z=1, y=4, x=5))

    def test_raises_on_time_mismatch(self) -> None:
        data = np.ones((3, 4, 5), dtype=np.float64)
        field = TimeDependentField.from_numpy(data, ("Y", "X"), ("center", "center"))
        with pytest.raises(ValueError, match="Expected size 10 along time axis but got 3"):
            field.validate_shape(SimulationSize(time=10, z=1, y=4, x=5))

    def test_raises_on_spatial_mismatch(self) -> None:
        data = np.ones((3, 4, 5), dtype=np.float64)
        field = TimeDependentField.from_numpy(data, ("Y", "X"), ("center", "center"))
        with pytest.raises(ValueError, match="Expected size 6 along axis X but got 5"):
            field.validate_shape(SimulationSize(time=3, z=1, y=4, x=6))

    def test_time_mismatch_is_checked_before_spatial_mismatch(self) -> None:
        # both the time and spatial dims are wrong; the time-axis error should surface first.
        data = np.ones((3, 4, 5), dtype=np.float64)
        field = TimeDependentField.from_numpy(data, ("Y", "X"), ("center", "center"))
        with pytest.raises(ValueError, match="Expected size 10 along time axis but got 3"):
            field.validate_shape(SimulationSize(time=10, z=1, y=4, x=6))


class TestSimulationSizeAxisSize:
    def test_returns_size_for_each_axis(self) -> None:
        size = SimulationSize(time=1, z=2, y=3, x=4)
        assert size.axis_size(ArrayAxis.Z) == 2
        assert size.axis_size(ArrayAxis.Y) == 3
        assert size.axis_size(ArrayAxis.X) == 4

    def test_invalid_axis_raises(self) -> None:
        size = SimulationSize(time=1, z=1, y=1, x=1)
        with pytest.raises(ValueError, match="Invalid axis"):
            size.axis_size("not-an-axis")  # type: ignore[arg-type]
