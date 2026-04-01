"""Tests for static (time-independent) output support."""

import functools

import numpy as np
import pytest
import zarr
import zarr.storage

from offline_particles.events import Event, SimulationState
from offline_particles.output import Output, ZarrOutputBuilder, ZarrOutputWriter
from offline_particles.particles import Particles, ParticlesView


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(nparticles: int = 5, property_values: dict | None = None) -> SimulationState:
    """Create a SimulationState with a single particle set for testing."""
    kwargs: dict = {"xidx": np.dtype(np.float64), "yidx": np.dtype(np.float64)}
    if property_values:
        for name, values in property_values.items():
            kwargs[name] = np.asarray(values).dtype

    particles = Particles(nparticles, **kwargs)

    if property_values:
        for name, values in property_values.items():
            particles[name][:] = np.asarray(values, dtype=particles[name].dtype)

    return SimulationState(
        time=np.float64(0.0),
        dt=np.float64(1.0),
        tidx=np.float64(0.0),
        iteration=0,
        wall_time=np.timedelta64(0, "ns"),
        particles={"particles": ParticlesView(particles)},
    )


def _make_output(property_name: str, dtype=np.float64) -> Output:
    """Create a simple Output object for testing."""
    return Output("particles", property_name, dtype=dtype)


def _make_builder(store: zarr.storage.StoreLike) -> ZarrOutputBuilder:
    """Create a ZarrOutputBuilder for testing."""
    return ZarrOutputBuilder("test_writer", store)


# ---------------------------------------------------------------------------
# ZarrOutputBuilder static output management
# ---------------------------------------------------------------------------


class TestZarrOutputBuilderStaticOutputs:
    def test_static_outputs_empty_initially(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        assert dict(builder.static_outputs) == {}

    def test_add_static_output(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        output = _make_output("density")
        builder.add_static_output("density", output)
        assert "density" in builder.static_outputs
        assert builder.static_outputs["density"] is output

    def test_add_static_output_duplicate_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        output = _make_output("density")
        builder.add_static_output("density", output)
        with pytest.raises(KeyError, match="density"):
            builder.add_static_output("density", output)

    def test_remove_static_output(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        output = _make_output("density")
        builder.add_static_output("density", output)
        builder.remove_static_output("density")
        assert "density" not in builder.static_outputs

    def test_remove_static_output_missing_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        with pytest.raises(KeyError, match="density"):
            builder.remove_static_output("density")

    def test_static_outputs_independent_from_outputs(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        output = _make_output("density")
        builder.add_output("x", _make_output("xidx"))
        builder.add_static_output("density", output)
        assert "density" not in builder.outputs
        assert "x" not in builder.static_outputs


# ---------------------------------------------------------------------------
# ZarrOutputWriter static output arrays
# ---------------------------------------------------------------------------


class TestZarrOutputWriterStaticOutputArrays:
    def test_build_creates_1d_static_array(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("density", _make_output("xidx"))

        writer = builder.build({"particles": 5})

        # The static output array should be 1D with shape (nparticles,)
        group = zarr.open_group(store, mode="r")
        assert "density" in group
        assert group["density"].shape == (5,)
        assert group["density"].ndim == 1

    def test_build_creates_2d_time_dependent_array(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("x", _make_output("xidx"))

        writer = builder.build({"particles": 5})

        group = zarr.open_group(store, mode="r")
        assert "x" in group
        assert group["x"].shape == (0, 5)
        assert group["x"].ndim == 2

    def test_static_outputs_property(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        output = _make_output("xidx")
        builder.add_static_output("density", output)

        writer = builder.build({"particles": 5})

        assert "density" in writer.static_outputs
        assert writer.static_outputs["density"] is output

    def test_static_outputs_not_in_outputs(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("density", _make_output("xidx"))

        writer = builder.build({"particles": 5})

        assert "density" not in writer.outputs


# ---------------------------------------------------------------------------
# ZarrOutputWriter.write_static_output
# ---------------------------------------------------------------------------


class TestZarrOutputWriterWriteStaticOutput:
    def test_write_static_output_writes_values(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("density", _make_output("xidx"))

        writer = builder.build({"particles": 5})

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = _make_state(5, {"xidx": values, "yidx": np.zeros(5)})

        writer.write_static_output("density", state)

        group = zarr.open_group(store, mode="r")
        np.testing.assert_array_equal(group["density"][:], values)

    def test_write_static_output_missing_key_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": 5})

        state = _make_state(5, {"xidx": np.zeros(5), "yidx": np.zeros(5)})

        with pytest.raises(KeyError, match="nonexistent"):
            writer.write_static_output("nonexistent", state)

    def test_write_static_output_does_not_affect_time_dependent_outputs(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("x", _make_output("xidx"))
        builder.add_static_output("density", _make_output("yidx"))

        writer = builder.build({"particles": 3})
        state = _make_state(3, {"xidx": np.ones(3), "yidx": np.array([10.0, 20.0, 30.0])})

        writer.write_static_output("density", state)

        group = zarr.open_group(store, mode="r")
        # time-dependent array should still be empty
        assert group["x"].shape[0] == 0


# ---------------------------------------------------------------------------
# create_static_events
# ---------------------------------------------------------------------------


class TestCreateStaticEvents:
    def test_create_static_events_empty_when_no_static_outputs(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": 5})
        events = writer.create_static_events()
        assert events == []

    def test_create_static_events_returns_one_event_per_output(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("density", _make_output("xidx"))
        builder.add_static_output("release_time", _make_output("yidx"))

        writer = builder.build({"particles": 5})
        events = writer.create_static_events()

        assert len(events) == 2

    def test_create_static_events_event_names(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("density", _make_output("xidx"))

        writer = builder.build({"particles": 5})
        events = writer.create_static_events()

        assert len(events) == 1
        assert events[0].name == "test_writer:static:density"

    def test_static_event_writes_data_when_invoked(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("density", _make_output("xidx"))

        writer = builder.build({"particles": 5})
        events = writer.create_static_events()

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = _make_state(5, {"xidx": values, "yidx": np.zeros(5)})

        # invoking the event should write the data
        events[0](state)

        group = zarr.open_group(store, mode="r")
        np.testing.assert_array_equal(group["density"][:], values)

    def test_create_events_does_not_include_static_events(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("x", _make_output("xidx"))
        builder.add_static_output("density", _make_output("yidx"))

        writer = builder.build({"particles": 5})
        recurring_events = writer.create_events()
        static_events = writer.create_static_events()

        recurring_names = {e.name for e in recurring_events}
        static_names = {e.name for e in static_events}

        # static events should not appear in recurring events and vice versa
        assert not (recurring_names & static_names)
        assert "test_writer:static:density" in static_names
        assert "test_writer:x" in recurring_names

    def test_static_array_dimension_names(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("density", _make_output("xidx"))

        writer = builder.build({"particles": 5})

        group = zarr.open_group(store, mode="r")
        dim_names = group["density"].metadata.dimension_names
        assert dim_names == ("particles",)
