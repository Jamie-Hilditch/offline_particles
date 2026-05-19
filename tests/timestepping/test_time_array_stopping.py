"""Tests for graceful stopping at the end of the time array."""

import numpy as np
import pytest

from offline_particles.fieldset import Fieldset
from offline_particles.simulation import SimulationBuilder
from offline_particles.timestepping import Clock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clock(time_array: np.ndarray, dt: float) -> Clock:
    """Create a Clock from a float64 time array and dt.

    Parameters
    ----------
        time_array (np.ndarray): A float64 array of times.
        dt (float): The time step size.

    Returns
    -------
        Clock: A Clock initialized with the given time array and dt.
    """
    return Clock(time_array, np.float64(dt))


def _make_fieldset() -> Fieldset:
    """Create a minimal Fieldset for testing.

    Returns
    -------
        Fieldset: A minimal Fieldset with dummy dimensions.
    """
    return Fieldset(1, 1, 1, 1)


def _make_builder(clock: Clock) -> SimulationBuilder:
    """Create a SimulationBuilder with an empty particle set list.

    Parameters
    ----------
        clock (Clock): The Clock to use for the simulation.

    Returns
    -------
        SimulationBuilder: A builder for creating a simulation with the given clock.
    """
    fieldset = _make_fieldset()
    builder = SimulationBuilder(clock, fieldset)
    return builder


# ---------------------------------------------------------------------------
# Clock.get_time_index boundary fix
# ---------------------------------------------------------------------------


class TestGetTimeIndexBoundary:
    def test_time_at_last_element_returns_last_index(self) -> None:
        """get_time_index should not raise an IndexError at time_array[-1].

        Previously, searchsorted returned len(time_array) for time == time_array[-1],
        causing an out-of-bounds access on time_array[idx + 1].
        """
        time_array = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        # This used to raise IndexError before the boundary fix
        idx = clock.get_time_index(np.float64(3.0))
        assert idx == pytest.approx(3.0)

    def test_time_at_first_element_returns_zero(self) -> None:
        time_array = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        idx = clock.get_time_index(np.float64(0.0))
        assert idx == pytest.approx(0.0)

    def test_time_in_last_interval(self) -> None:
        """Interpolation in the last interval should work correctly."""
        time_array = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        # midpoint of last interval [2, 3] => index 2.5
        idx = clock.get_time_index(np.float64(2.5))
        assert idx == pytest.approx(2.5)

    def test_time_array_with_two_elements(self) -> None:
        """Boundary fix should work for the minimum-length time array."""
        time_array = np.array([0.0, 1.0], dtype=np.float64)
        clock = _make_clock(time_array, 0.5)
        idx = clock.get_time_index(np.float64(1.0))
        assert idx == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Default time_stop initialised from time array
# ---------------------------------------------------------------------------


class TestDefaultTimeStop:
    def test_default_time_stop_is_time_array_end_for_forward(self) -> None:
        """A forward simulation should have time_stop set to time_array[-1] by default."""
        time_array = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        clock = _make_clock(time_array, 1.0)
        sim = _make_builder(clock).build_simulation()
        assert sim.time_stop == pytest.approx(3.0)

    def test_default_time_stop_is_time_array_start_for_backward(self) -> None:
        """A backward simulation should have time_stop set to time_array[0] by default."""
        time_array = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        clock = _make_clock(time_array, -1.0)
        clock.set_time(time_array[-1])
        sim = _make_builder(clock).build_simulation()
        assert sim.time_stop == pytest.approx(0.0)

    def test_default_time_stop_can_be_overridden(self) -> None:
        """The default time_stop should be overridable with set_time_stop."""
        time_array = np.arange(0.0, 10.0, dtype=np.float64)
        clock = _make_clock(time_array, 1.0)
        sim = _make_builder(clock).build_simulation()
        sim.set_time_stop(np.float64(5.0))
        assert sim.time_stop == pytest.approx(5.0)

    def test_default_time_stop_can_be_cleared(self) -> None:
        """The default time_stop can be cleared by passing None to set_time_stop."""
        time_array = np.arange(0.0, 10.0, dtype=np.float64)
        clock = _make_clock(time_array, 1.0)
        sim = _make_builder(clock).build_simulation()
        sim.set_time_stop(None)
        assert sim.time_stop is None


# ---------------------------------------------------------------------------
# Forward simulation stops gracefully at end of time array
# ---------------------------------------------------------------------------


class TestForwardSimulationTimeArrayStop:
    def test_run_stops_at_end_of_time_array_no_explicit_stop(self) -> None:
        """Default stop at end of time array for forward simulation.

        A forward simulation should stop at the end of the time array
        even when no explicit stopping condition is set.
        """
        time_array = np.arange(0.0, 5.0, dtype=np.float64)  # [0, 1, 2, 3, 4]
        clock = _make_clock(time_array, 1.0)
        sim = _make_builder(clock).build_simulation()

        sim.run()
        assert sim.time <= time_array[-1]

    def test_run_does_not_raise_at_end_of_time_array(self) -> None:
        """Reaching the end of the time array should not raise an error."""
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 1.0)
        sim = _make_builder(clock).build_simulation()

        # Should complete without IndexError or ValueError
        sim.run()
        assert sim.time == pytest.approx(2.0)

    def test_run_valid_without_explicit_stopping_condition(self) -> None:
        """Default time stop is valid stopping condition.

        A simulation with no explicitly user-set stopping conditions should not
        raise 'No valid stopping condition' because the default time_stop is used.
        """
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, 1.0)
        sim = _make_builder(clock).build_simulation()

        # This must not raise ValueError about missing stopping conditions
        sim.run()


# ---------------------------------------------------------------------------
# Backward simulation stops gracefully at start of time array
# ---------------------------------------------------------------------------


class TestBackwardSimulationTimeArrayStop:
    def test_run_stops_at_start_of_time_array_no_explicit_stop(self) -> None:
        """Backward simulation stops at start of time array.

        A backward simulation should stop at the start of the time array
        even when no explicit stopping condition is set.
        """
        time_array = np.arange(0.0, 5.0, dtype=np.float64)  # [0, 1, 2, 3, 4]
        clock = _make_clock(time_array, -1.0)
        clock.set_time(time_array[-1])
        sim = _make_builder(clock).build_simulation()

        sim.run()
        assert sim.time >= time_array[0]

    def test_run_does_not_raise_for_backward_simulation(self) -> None:
        """Can reach the start of the time array in a backward simulation.

        Reaching the start of the time array in a backward simulation should
        not raise an error.
        """
        time_array = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        clock = _make_clock(time_array, -1.0)
        clock.set_time(time_array[-1])
        sim = _make_builder(clock).build_simulation()

        sim.run()
        assert sim.time == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Explicit stopping conditions still work alongside the default time_stop
# ---------------------------------------------------------------------------


class TestExplicitStoppingConditionsWithTimeArrayBound:
    def test_explicit_iteration_stop_takes_precedence(self) -> None:
        """An explicit iteration stop should still work."""
        time_array = np.arange(0.0, 10.0, dtype=np.float64)
        clock = _make_clock(time_array, 1.0)
        sim = _make_builder(clock).build_simulation()
        sim.set_stopping_conditions(iteration=3)

        sim.run()
        assert sim.iteration == 3

    def test_explicit_time_stop_overrides_default(self) -> None:
        """Explicit time stop overrides the default time_stop from the time array.

        An explicit time stop before the end of the time array should override
        the default time_stop.
        """
        time_array = np.arange(0.0, 10.0, dtype=np.float64)
        clock = _make_clock(time_array, 1.0)
        sim = _make_builder(clock).build_simulation()
        sim.set_stopping_conditions(time=np.float64(5.0))

        sim.run()
        assert sim.time == pytest.approx(5.0)
