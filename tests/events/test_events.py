"""Tests for the events module (Event, SimulationState)."""

import numpy as np
import pytest

from offline_particles.events import Event, SimulationState
from offline_particles.particles import Particles, ParticlesView

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(time: float = 0.0, iteration: int = 0) -> SimulationState:
    particles = Particles(3, {"x": np.dtype(np.float64)})
    view = ParticlesView(particles)
    return SimulationState(
        time=np.float64(time),
        dt=np.float64(1.0),
        tidx=np.float64(time),
        iteration=iteration,
        wall_time=np.timedelta64(0, "s"),
        particles={"particles": view},
    )


# ---------------------------------------------------------------------------
# SimulationState
# ---------------------------------------------------------------------------


class TestSimulationState:
    def test_construction(self) -> None:
        state = _make_state(1.0, 5)
        assert state.time == np.float64(1.0)
        assert state.dt == np.float64(1.0)
        assert state.iteration == 5

    def test_particles_mapping(self) -> None:
        state = _make_state()
        assert "particles" in state.particles

    def test_is_frozen_dataclass(self) -> None:
        state = _make_state()
        with pytest.raises((AttributeError, TypeError)):
            state.iteration = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Event construction and calling
# ---------------------------------------------------------------------------


class TestEventConstruction:
    def test_name_property(self) -> None:
        def noop(state: SimulationState) -> None:
            pass

        event = Event("my_event", noop)
        assert event.name == "my_event"

    def test_no_kernels_by_default(self) -> None:
        def noop(state: SimulationState) -> None:
            pass

        event = Event("e", noop)
        assert dict(event.kernels) == {}

    def test_str_contains_name(self) -> None:
        def noop(state: SimulationState) -> None:
            pass

        event = Event("my_event", noop)
        assert "my_event" in str(event)


class TestEventCalling:
    def test_invokes_function(self) -> None:
        called = []

        def record(state: SimulationState) -> None:
            called.append(state.iteration)

        event = Event("e", record)
        state = _make_state(iteration=7)
        event(state)
        assert called == [7]

    def test_invokes_function_with_correct_state(self) -> None:
        received = []

        def capture(state: SimulationState) -> None:
            received.append(state)

        event = Event("e", capture)
        state = _make_state(time=3.0, iteration=2)
        event(state)
        assert len(received) == 1
        assert received[0] is state

    def test_callable_multiple_times(self) -> None:
        counter = [0]

        def increment(state: SimulationState) -> None:
            counter[0] += 1

        event = Event("e", increment)
        state = _make_state()
        event(state)
        event(state)
        event(state)
        assert counter[0] == 3


class TestEventKernels:
    def test_kernels_are_stored(self) -> None:
        from offline_particles.kernels import BoundKernel, ParticleKernel, ParticlePropertyDeclaration

        def fn(pp, sc, fd):
            pass

        kernel = ParticleKernel(fn, [ParticlePropertyDeclaration("x", np.float64)])
        bound = BoundKernel(kernel)

        def noop(state: SimulationState) -> None:
            pass

        event = Event("e", noop, particles=(bound,))
        assert "particles" in event.kernels
        assert bound in event.kernels["particles"]
