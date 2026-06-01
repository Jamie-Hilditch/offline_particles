"""Tests for static (time-independent) output support."""

import numpy as np
import pytest
import zarr
import zarr.storage

from offline_particles.events import SimulationState
from offline_particles.output import Output, ZarrOutputBuilder
from tests.output._helpers import make_particles_view

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(nparticles: int = 5, property_values: dict | None = None) -> SimulationState:
    """Create a SimulationState with a single particle set for testing.

    Parameters
    ----------
    nparticles : int
        The number of particles in the particle set.
    property_values : dict, optional
        A dictionary mapping property names to their values. If provided, these values will be set in
        the particle set.

    Returns
    -------
    SimulationState
        A SimulationState object with the specified number of particles and property values.
    """
    return SimulationState(
        time=np.float64(0.0),
        dt=np.float64(1.0),
        tidx=np.float64(0.0),
        iteration=0,
        wall_time=np.timedelta64(0, "ns"),
        particles={"particles": make_particles_view(nparticles, property_values)},
    )


def _make_output(property_name: str, dtype=np.float64) -> Output:
    """Create a simple Output object for testing.

    Parameters
    ----------
    property_name : str
        The name of the property to output.
    dtype : data-type, optional
        The data type of the output array. Default is np.float64.

    Returns
    -------
    Output
        An Output object configured for the specified property and data type.
    """
    return Output(property_name, dtype=dtype)


def _make_builder(store: zarr.storage.StoreLike) -> ZarrOutputBuilder:
    """Create a ZarrOutputBuilder for testing.

    Parameters
    ----------
    store : zarr.storage.StoreLike
        The storage backend for the Zarr array.

    Returns
    -------
    ZarrOutputBuilder
        A ZarrOutputBuilder instance for testing.
    """
    return ZarrOutputBuilder("test_writer", store)


# ---------------------------------------------------------------------------
# ZarrOutputBuilder static output management
# ---------------------------------------------------------------------------


class TestZarrOutputBuilderStaticOutputs:
    def test_static_outputs_empty_initially(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        assert list(builder.static_outputs) == []

    def test_add_static_output(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        output = _make_output("density")
        builder.add_static_output("particles", "density", output)
        assert ("particles", "density") in dict(builder.static_outputs)
        assert dict(builder.static_outputs)[("particles", "density")] is output

    def test_add_static_output_duplicate_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        output = _make_output("density")
        builder.add_static_output("particles", "density", output)
        with pytest.raises(KeyError, match="density"):
            builder.add_static_output("particles", "density", output)

    def test_remove_static_output(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        output = _make_output("density")
        builder.add_static_output("particles", "density", output)
        builder.remove_static_output("particles", "density")
        assert ("particles", "density") not in dict(builder.static_outputs)

    def test_remove_static_output_missing_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        with pytest.raises(KeyError, match="density"):
            builder.remove_static_output("particles", "density")

    def test_static_outputs_independent_from_outputs(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        output = _make_output("density")
        builder.add_output("particles", "x", _make_output("xidx"))
        builder.add_static_output("particles", "density", output)
        assert ("particles", "density") not in dict(builder.outputs)
        assert ("particles", "x") not in dict(builder.static_outputs)

    def test_add_output_clashes_with_static_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("particles", "density", _make_output("xidx"))
        with pytest.raises(KeyError, match="density"):
            builder.add_output("particles", "density", _make_output("yidx"))

    def test_add_static_output_clashes_with_output_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("particles", "density", _make_output("xidx"))
        with pytest.raises(KeyError, match="density"):
            builder.add_static_output("particles", "density", _make_output("yidx"))


# ---------------------------------------------------------------------------
# ZarrOutputWriter static output arrays
# ---------------------------------------------------------------------------


class TestZarrOutputWriterStaticOutputArrays:
    def test_build_creates_1d_static_array(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("particles", "density", _make_output("xidx"))

        builder.build({"particles": make_particles_view(5)})

        # The static output array should be 1D with shape (nparticles,)
        group = zarr.open_group(store, mode="r")
        assert "density" in group["particles"]
        arr = group["particles"]["density"]  # type: ignore[invalid-argument-type]
        assert isinstance(arr, zarr.Array)
        assert arr.shape == (5,)
        assert arr.ndim == 1

    def test_build_creates_2d_time_dependent_array(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("particles", "x", _make_output("xidx"))

        builder.build({"particles": make_particles_view(5)})

        group = zarr.open_group(store, mode="r")
        assert "x" in group["particles"]
        arr = group["particles"]["x"]  # type: ignore[invalid-argument-type]
        assert isinstance(arr, zarr.Array)
        assert arr.shape == (0, 5)
        assert arr.ndim == 2

    def test_static_outputs_property(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        output = _make_output("xidx")
        builder.add_static_output("particles", "density", output)

        writer = builder.build({"particles": make_particles_view(5)})

        assert ("particles", "density") in dict(writer.static_outputs)
        assert dict(writer.static_outputs)[("particles", "density")] is output

    def test_static_outputs_not_in_outputs(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("particles", "density", _make_output("xidx"))

        writer = builder.build({"particles": make_particles_view(5)})

        assert ("particles", "density") not in dict(writer.outputs)


# ---------------------------------------------------------------------------
# ZarrOutputWriter.write_static_output
# ---------------------------------------------------------------------------


class TestZarrOutputWriterWriteStaticOutput:
    def test_write_static_output_writes_values(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("particles", "density", _make_output("xidx"))

        writer = builder.build({"particles": make_particles_view(5)})

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = _make_state(5, {"xidx": values, "yidx": np.zeros(5)})

        writer.write_static_output("particles", "density", state)

        group = zarr.open_group(store, mode="r")
        arr = group["particles"]["density"]  # type: ignore[invalid-argument-type]
        assert isinstance(arr, zarr.Array)
        np.testing.assert_array_equal(arr[:], values)

    def test_write_static_output_missing_key_raises(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": make_particles_view(5)})

        state = _make_state(5, {"xidx": np.zeros(5), "yidx": np.zeros(5)})

        with pytest.raises(KeyError, match="nonexistent"):
            writer.write_static_output("particles", "nonexistent", state)

    def test_write_static_output_does_not_affect_time_dependent_outputs(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("particles", "x", _make_output("xidx"))
        builder.add_static_output("particles", "density", _make_output("yidx"))

        writer = builder.build({"particles": make_particles_view(3)})
        state = _make_state(3, {"xidx": np.ones(3), "yidx": np.array([10.0, 20.0, 30.0])})

        writer.write_static_output("particles", "density", state)

        group = zarr.open_group(store, mode="r")
        # time-dependent array should still be empty
        x_arr = group["particles"]["x"]  # type: ignore[invalid-argument-type]
        assert isinstance(x_arr, zarr.Array)
        assert x_arr.shape[0] == 0


# ---------------------------------------------------------------------------
# create_static_events
# ---------------------------------------------------------------------------


class TestCreateStaticOutputEvents:
    def test_create_static_output_events_empty_when_no_static_outputs(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        writer = builder.build({"particles": make_particles_view(5)})
        events = writer.create_static_output_events()
        assert events == []

    def test_create_static_output_events_returns_one_event_per_output(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("particles", "density", _make_output("xidx"))
        builder.add_static_output("particles", "release_time", _make_output("yidx"))

        writer = builder.build({"particles": make_particles_view(5)})
        events = writer.create_static_output_events()

        assert len(events) == 2

    def test_create_static_output_events_event_names(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("particles", "density", _make_output("xidx"))

        writer = builder.build({"particles": make_particles_view(5)})
        events = writer.create_static_output_events()

        assert len(events) == 1
        assert events[0].name == "test_writer:particles:density"

    def test_static_event_writes_data_when_invoked(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("particles", "density", _make_output("xidx"))

        writer = builder.build({"particles": make_particles_view(5)})
        events = writer.create_static_output_events()

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = _make_state(5, {"xidx": values, "yidx": np.zeros(5)})

        # invoking the event should write the data
        events[0](state)

        group = zarr.open_group(store, mode="r")
        arr = group["particles"]["density"]  # type: ignore[invalid-argument-type]
        assert isinstance(arr, zarr.Array)
        np.testing.assert_array_equal(arr[:], values)

    def test_create_output_events_does_not_include_static_events(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_output("particles", "x", _make_output("xidx"))
        builder.add_static_output("particles", "density", _make_output("yidx"))

        writer = builder.build({"particles": make_particles_view(5)})
        recurring_events = writer.create_output_events()
        static_events = writer.create_static_output_events()

        recurring_names = {e.name for e in recurring_events}
        static_names = {e.name for e in static_events}

        # static events should not appear in recurring events and vice versa
        assert not (recurring_names & static_names)
        assert "test_writer:particles:density" in static_names
        assert "test_writer:particles:x" in recurring_names

    def test_static_array_dimension_names(self) -> None:
        store = zarr.storage.MemoryStore()
        builder = _make_builder(store)
        builder.add_static_output("particles", "density", _make_output("xidx"))

        builder.build({"particles": make_particles_view(5)})

        group = zarr.open_group(store, mode="r")
        density_arr = group["particles"]["density"]  # type: ignore[invalid-argument-type]
        assert isinstance(density_arr, zarr.Array)
        dim_names = getattr(density_arr.metadata, "dimension_names", None)
        assert dim_names == ("particles",)
