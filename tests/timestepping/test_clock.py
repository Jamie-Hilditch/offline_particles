"""Tests for the Clock class (validation, construction, methods)."""

import numpy as np
import pytest

from offline_particles.timestepping import Clock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clock(time_array: np.ndarray, dt: float) -> Clock:
    return Clock(time_array, np.float64(dt))


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestClockConstructionValidation:
    def test_rejects_2d_time_array(self) -> None:
        time_array = np.ones((3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="1D"):
            _make_clock(time_array, 1.0)

    def test_rejects_single_element_time_array(self) -> None:
        time_array = np.array([0.0], dtype=np.float64)
        with pytest.raises(ValueError, match="at least 2"):
            _make_clock(time_array, 1.0)

    def test_rejects_non_increasing_time_array(self) -> None:
        time_array = np.array([0.0, 2.0, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="strictly increasing"):
            _make_clock(time_array, 1.0)

    def test_rejects_constant_time_array(self) -> None:
        time_array = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="strictly increasing"):
            _make_clock(time_array, 1.0)

    def test_rejects_dimensional_dt_with_default_time_unit(self) -> None:
        time_array = np.array([0.0, 1.0], dtype=np.float64)
        dt = np.timedelta64(1, "s")
        with pytest.raises(TypeError, match="time_unit"):
            Clock(time_array, dt)

    def test_rejects_zero_time_unit(self) -> None:
        time_array = np.array([0.0, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="time_unit must be positive"):
            Clock(time_array, np.float64(1.0), time_unit=np.float64(0.0))

    def test_rejects_negative_time_unit(self) -> None:
        time_array = np.array([0.0, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="time_unit must be positive"):
            Clock(time_array, np.float64(1.0), time_unit=np.float64(-1.0))

    def test_accepts_two_element_time_array(self) -> None:
        time_array = np.array([0.0, 1.0], dtype=np.float64)
        clock = _make_clock(time_array, np.float64(0.5))
        assert clock is not None

    def test_accepts_python_float_dt_without_time_unit(self) -> None:
        time_array = np.array([0.0, 1.0], dtype=np.float64)
        clock = Clock(time_array, 0.5)
        assert clock.dt == np.float64(0.5)

    def test_accepts_numpy_float_scalar_dt_without_time_unit(self) -> None:
        time_array = np.array([0.0, 1.0], dtype=np.float64)
        clock = Clock(time_array, np.float64(0.5))
        assert clock.time_unit == np.float64(1.0)

    def test_accepts_dimensional_time_array_dt_and_time_unit(self) -> None:
        time_array = np.array(
            ["2000-01-01T00:00:00", "2000-01-01T01:00:00"],
            dtype="datetime64[s]",
        )
        dt = np.timedelta64(30, "m")
        time_unit = np.timedelta64(1, "m")
        clock = Clock(time_array, dt, time_unit=time_unit)
        np.testing.assert_array_equal(clock.time_array, time_array)
        assert clock.dt == dt
        assert clock.time_unit == time_unit

    def test_accepts_mixed_compatible_timedelta_units(self) -> None:
        time_array = np.array(
            ["2000-01-01T00:00:00", "2000-01-01T02:00:00"],
            dtype="datetime64[s]",
        )
        dt = np.timedelta64(1500, "ms")
        time_unit = np.timedelta64(1, "s")
        clock = Clock(time_array, dt, time_unit=time_unit)
        np.testing.assert_array_equal(clock.time_array, time_array)
        assert dt == np.timedelta64(1500, "ms")
        assert clock.dt == np.timedelta64(1, "s")
        assert clock.dt != dt
        assert clock.time_unit == time_unit


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestClockProperties:
    def test_dt_property(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        assert clock.dt == pytest.approx(0.5)

    def test_time_unit_property(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = Clock(time_array, np.float64(0.5), time_unit=np.float64(2.0))
        assert clock.time_unit == np.float64(2.0)

    def test_forward_in_time_for_positive_dt(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 1.0)
        assert clock.forward_in_time

    def test_forward_in_time_for_negative_dt(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, -1.0)
        assert not clock.forward_in_time

    def test_first_time_forward(self) -> None:
        time_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        assert clock.first_time == pytest.approx(1.0)

    def test_first_time_backward(self) -> None:
        time_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        clock = _make_clock(time_array, -0.5)
        assert clock.first_time == pytest.approx(3.0)

    def test_final_time_forward(self) -> None:
        time_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        assert clock.final_time == pytest.approx(3.0)

    def test_final_time_backward(self) -> None:
        time_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        clock = _make_clock(time_array, -0.5)
        assert clock.final_time == pytest.approx(1.0)

    def test_time_array_property(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        np.testing.assert_array_equal(clock.time_array, time_array)

    def test_iteration_property_initial(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        assert clock.iteration == 0

    def test_tinfo_property(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        tinfo = clock.tinfo
        assert tinfo.time == clock.time
        assert tinfo.tidx == clock.tidx
        assert tinfo.iteration == clock.iteration


# ---------------------------------------------------------------------------
# set_dt
# ---------------------------------------------------------------------------


class TestClockSetDt:
    def test_set_dt_forward(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        clock.set_dt(np.float64(0.25))
        assert clock.dt == pytest.approx(0.25)

    def test_set_dt_rejects_negative_for_forward(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        with pytest.raises(ValueError, match="positive"):
            clock.set_dt(np.float64(-0.5))

    def test_set_dt_rejects_positive_for_backward(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, -0.5)
        with pytest.raises(ValueError, match="negative"):
            clock.set_dt(np.float64(0.5))

    def test_set_dt_rejects_zero_for_forward(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        with pytest.raises(ValueError, match="positive"):
            clock.set_dt(np.float64(0.0))


# ---------------------------------------------------------------------------
# set_time
# ---------------------------------------------------------------------------


class TestClockSetTime:
    def test_set_time_valid(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        clock.set_time(np.float64(1.5))
        assert clock.time == pytest.approx(1.5)

    def test_set_time_out_of_bounds_raises(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        with pytest.raises(ValueError, match="out of bounds"):
            clock.set_time(np.float64(5.0))

    def test_set_time_updates_tidx(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        clock.set_time(np.float64(1.0))
        assert clock.tidx == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# set_iteration
# ---------------------------------------------------------------------------


class TestClockSetIteration:
    def test_set_iteration_valid(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        clock.set_iteration(5)
        assert clock.iteration == 5

    def test_set_iteration_negative_raises(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        with pytest.raises(ValueError, match="non-negative"):
            clock.set_iteration(-1)


# ---------------------------------------------------------------------------
# advance_time
# ---------------------------------------------------------------------------


class TestClockAdvanceTime:
    def test_advance_time_forward(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        clock = _make_clock(time_array, 1.0)
        clock.advance_time()
        assert clock.time == pytest.approx(1.0)
        assert clock.iteration == 1

    def test_advance_time_backward(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        clock = _make_clock(time_array, -1.0)
        clock.advance_time()
        assert clock.time == pytest.approx(2.0)
        assert clock.iteration == 1

    def test_advance_time_updates_tidx(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        clock = _make_clock(time_array, 1.0)
        clock.advance_time()
        assert clock.tidx == pytest.approx(1.0)

    def test_advance_time_increments_iteration(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        for _ in range(3):
            clock.advance_time()
        assert clock.iteration == 3


# ---------------------------------------------------------------------------
# get_time_index (interpolation)
# ---------------------------------------------------------------------------


class TestClockGetTimeIndex:
    def test_midpoint_interpolation(self) -> None:
        time_array = np.array([0.0, 2.0, 4.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        idx = clock.get_time_index(np.float64(1.0))
        assert idx == pytest.approx(0.5)

    def test_out_of_bounds_raises(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        with pytest.raises(ValueError, match="out of bounds"):
            clock.get_time_index(np.float64(-0.1))

    def test_out_of_bounds_high_raises(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        with pytest.raises(ValueError, match="out of bounds"):
            clock.get_time_index(np.float64(2.1))
