"""Tests for time-dependent (recurring) output: write_output, write_time, finalise_write_round."""

import numpy as np
import pytest
import zarr
import zarr.storage

from offline_particles.events import SimulationState
from offline_particles.output import Output, ZarrOutputBuilder
from offline_particles.particles import Particles, ParticlesView

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    nparticles: int = 3,
    property_values: dict | None = None,
    time: float = 0.0,
    particle_set: str = "particles",
) -> SimulationState:
    """Create a SimulationState with a single particle set for testing.

    Parameters
    ----------
        nparticles (int): The number of particles in the set.
        property_values (dict | None): Optional dictionary of property names and their values.
        time (float): The simulation time.
        particle_set (str): The name of the particle set.

    Returns
    -------
        SimulationState: A SimulationState containing the specified particle set.
    """
    kwargs: dict = {"xidx": np.dtype(np.float64), "yidx": np.dtype(np.float64)}
    if property_values:
        for name, values in property_values.items():
            kwargs[name] = np.asarray(values).dtype

    p = Particles(nparticles, **kwargs)

    if property_values:
        for name, values in property_values.items():
            p[name][:] = np.asarray(values, dtype=p[name].dtype)

    return SimulationState(
        time=np.float64(time),
        dt=np.float64(1.0),
        tidx=np.float64(0.0),
        iteration=0,
        wall_time=np.timedelta64(0, "ns"),
        particles={particle_set: ParticlesView(p)},
    )


def _make_multi_set_state(
    nparticles_ps1: int = 3,
    nparticles_ps2: int = 2,
    values_ps1: list | None = None,
    values_ps2: list | None = None,
    time: float = 0.0,
) -> SimulationState:
    """Create a SimulationState with two particle sets.

    Parameters
    ----------
        nparticles_ps1 (int): Number of particles in the first particle set.
        nparticles_ps2 (int): Number of particles in the second particle set.
        values_ps1 (list | None): Optional list of values for the first particle set's 'xidx' property.
        values_ps2 (list | None): Optional list of values for the second particle set's 'xidx' property.
        time (float): The simulation time.

    Returns
    -------
        SimulationState: A SimulationState containing two particle sets.
    """
    p1 = Particles(nparticles_ps1, xidx=np.dtype(np.float64))
    if values_ps1 is not None:
        p1["xidx"][:] = np.asarray(values_ps1, dtype=np.float64)

    p2 = Particles(nparticles_ps2, xidx=np.dtype(np.float64))
    if values_ps2 is not None:
        p2["xidx"][:] = np.asarray(values_ps2, dtype=np.float64)

    return SimulationState(
        time=np.float64(time),
        dt=np.float64(1.0),
        tidx=np.float64(0.0),
        iteration=0,
        wall_time=np.timedelta64(0, "ns"),
        particles={"ps1": ParticlesView(p1), "ps2": ParticlesView(p2)},
    )


def _make_builder(store: zarr.storage.StoreLike) -> ZarrOutputBuilder:
    return ZarrOutputBuilder("test_writer", store)


# ---------------------------------------------------------------------------
# ZarrOutputBuilder.build
# ---------------------------------------------------------------------------


class TestZarrOutputBuilderBuild:
    def test_build_creates_time_array(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.build({"particles": 5})

        group = zarr.open_group(store, mode="r")
        assert "time" in group["particles"]
        arr = group["particles"]["time"]  # type: ignore[invalid-argument-type]
        assert isinstance(arr, zarr.Array)
        assert arr.shape == (0,)

    def test_build_creates_groups_for_each_particle_set(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.build({"ps1": 3, "ps2": 5})

        group = zarr.open_group(store, mode="r")
        assert "time" in group["ps1"]
        assert "time" in group["ps2"]

    def test_build_output_for_unknown_particle_set_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("unknown_ps", "x", Output("xidx"))
        with pytest.raises(KeyError, match="unknown_ps"):
            builder.build({"particles": 5})

    def test_build_static_output_for_unknown_particle_set_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("unknown_ps", "density", Output("xidx"))
        with pytest.raises(KeyError, match="unknown_ps"):
            builder.build({"particles": 5})

    def test_time_array_dimension_name(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.build({"particles": 5})

        group = zarr.open_group(store, mode="r")
        time_arr = group["particles"]["time"]  # type: ignore[invalid-argument-type]
        assert isinstance(time_arr, zarr.Array)
        dim_names = getattr(time_arr.metadata, "dimension_names", None)
        assert dim_names == ("time",)

    def test_output_array_dimension_names(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("particles", "x", Output("xidx"))
        builder.build({"particles": 5})

        group = zarr.open_group(store, mode="r")
        x_arr = group["particles"]["x"]  # type: ignore[invalid-argument-type]
        assert isinstance(x_arr, zarr.Array)
        dim_names = getattr(x_arr.metadata, "dimension_names", None)
        assert dim_names == ("time", "particles")

    def test_remove_output(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("particles", "x", Output("xidx"))
        builder.remove_output("particles", "x")
        assert ("particles", "x") not in dict(builder.outputs)

    def test_remove_output_missing_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        with pytest.raises(KeyError, match="x"):
            builder.remove_output("particles", "x")


# ---------------------------------------------------------------------------
# ZarrOutputWriter.write_time
# ---------------------------------------------------------------------------


class TestZarrOutputWriterWriteTime:
    def test_write_time_appends_to_time_array(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": 3})

        state = _make_state(time=5.0)
        writer.write_time(state)

        group = zarr.open_group(store, mode="r")
        np.testing.assert_array_equal(group["particles"]["time"][:], [5.0])  # type: ignore[invalid-argument-type]

    def test_write_time_appends_multiple_values(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": 3})

        for t in [0.0, 1.0, 2.0]:
            state = _make_state(time=t)
            writer.write_time(state)

        group = zarr.open_group(store, mode="r")
        np.testing.assert_array_equal(group["particles"]["time"][:], [0.0, 1.0, 2.0])  # type: ignore[invalid-argument-type]

    def test_write_time_writes_to_all_particle_set_groups(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("ps1", "x", Output("xidx"))
        builder.add_output("ps2", "x", Output("xidx"))
        writer = builder.build({"ps1": 3, "ps2": 2})

        state = _make_multi_set_state(time=7.0)
        writer.write_time(state)

        group = zarr.open_group(store, mode="r")
        np.testing.assert_array_equal(group["ps1"]["time"][:], [7.0])  # type: ignore[invalid-argument-type]
        np.testing.assert_array_equal(group["ps2"]["time"][:], [7.0])  # type: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# ZarrOutputWriter.write_output
# ---------------------------------------------------------------------------


class TestZarrOutputWriterWriteOutput:
    def test_write_output_creates_new_row(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("particles", "x", Output("xidx"))
        writer = builder.build({"particles": 3})

        values = np.array([1.0, 2.0, 3.0])
        state = _make_state(3, {"xidx": values})
        writer.write_output("particles", "x", state)

        group = zarr.open_group(store, mode="r")
        x_arr = group["particles"]["x"]  # type: ignore[invalid-argument-type]
        assert isinstance(x_arr, zarr.Array)
        assert x_arr.shape == (1, 3)
        np.testing.assert_array_equal(x_arr[0, :], values)

    def test_write_output_appends_across_timesteps(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("particles", "x", Output("xidx"))
        writer = builder.build({"particles": 3})

        for i in range(3):
            values = np.array([float(i)] * 3)
            state = _make_state(3, {"xidx": values})
            writer.write_output("particles", "x", state)

        group = zarr.open_group(store, mode="r")
        x_arr = group["particles"]["x"]  # type: ignore[invalid-argument-type]
        assert x_arr.shape == (3, 3)  # type: ignore[possibly-missing-attribute]
        for i in range(3):
            np.testing.assert_array_equal(x_arr[i, :], [float(i)] * 3)  # type: ignore[invalid-argument-type]

    def test_write_output_missing_key_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": 3})

        state = _make_state(3)
        with pytest.raises(KeyError, match="nonexistent"):
            writer.write_output("particles", "nonexistent", state)

    def test_write_output_multiple_particle_sets(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("ps1", "x", Output("xidx"))
        builder.add_output("ps2", "x", Output("xidx"))
        writer = builder.build({"ps1": 3, "ps2": 2})

        state = _make_multi_set_state(
            values_ps1=[1.0, 2.0, 3.0],
            values_ps2=[4.0, 5.0],
        )
        writer.write_output("ps1", "x", state)
        writer.write_output("ps2", "x", state)

        group = zarr.open_group(store, mode="r")
        np.testing.assert_array_equal(group["ps1"]["x"][0, :], [1.0, 2.0, 3.0])  # type: ignore[invalid-argument-type]
        np.testing.assert_array_equal(group["ps2"]["x"][0, :], [4.0, 5.0])  # type: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# ZarrOutputWriter.finalise_write_round
# ---------------------------------------------------------------------------


class TestZarrOutputWriterFinaliseWriteRound:
    def test_finalise_increments_count(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": 3})

        state = _make_state()
        writer.write_time(state)
        writer.finalise_write_round(state)

        # A second write_time + finalise should work without error
        writer.write_time(state)
        writer.finalise_write_round(state)

    def test_finalise_raises_if_time_not_written(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": 3})

        state = _make_state()
        with pytest.raises(RuntimeError, match="Time output"):
            writer.finalise_write_round(state)

    def test_finalise_raises_if_output_not_written(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("particles", "x", Output("xidx"))
        writer = builder.build({"particles": 3})

        state = _make_state()
        writer.write_time(state)

        with pytest.raises(RuntimeError, match="Output"):
            writer.finalise_write_round(state)


# ---------------------------------------------------------------------------
# ZarrOutputWriter.create_output_events
# ---------------------------------------------------------------------------


class TestCreateOutputEvents:
    def test_create_output_events_includes_time_event(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": 3})

        events = writer.create_output_events()
        event_names = [e.name for e in events]
        assert "test_writer:time" in event_names

    def test_create_output_events_includes_finalise_event(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": 3})

        events = writer.create_output_events()
        event_names = [e.name for e in events]
        assert "test_writer:finalise" in event_names

    def test_create_output_events_includes_output_event(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("particles", "x", Output("xidx"))
        writer = builder.build({"particles": 3})

        events = writer.create_output_events()
        event_names = [e.name for e in events]
        assert "test_writer:particles:x" in event_names

    def test_create_output_events_minimum_count(self) -> None:
        """With no outputs, there should be at least time and finalise events."""
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": 3})

        events = writer.create_output_events()
        assert len(events) >= 2

    def test_output_event_writes_data_when_invoked(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("particles", "x", Output("xidx"))
        writer = builder.build({"particles": 3})

        events = writer.create_output_events()
        values = np.array([10.0, 20.0, 30.0])
        state = _make_state(3, {"xidx": values})

        # invoke all events in order
        for event in events:
            event(state)

        group = zarr.open_group(store, mode="r")
        x_arr = group["particles"]["x"]  # type: ignore[invalid-argument-type]
        assert isinstance(x_arr, zarr.Array)
        np.testing.assert_array_equal(x_arr[0, :], values)

    def test_event_name_method(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": 3})

        assert writer.event_name("particles", "x") == "test_writer:particles:x"
        assert writer.event_name("ps2", "density") == "test_writer:ps2:density"
