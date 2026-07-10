"""Tests for output dtype inference from particle properties."""

import numpy as np
import zarr

from offline_particles.output import Output


class TestZarrOutputBuilderDtypeInference:
    def test_time_dependent_output_uses_particle_property_dtype_when_dtype_is_none(
        self, zarr_store, make_output_builder, make_particles_view
    ) -> None:
        store = zarr_store
        builder = make_output_builder(store)
        builder.add_output("particles", "x", Output("mass", dtype=None))

        builder.build({"particles": make_particles_view(4, {"mass": np.array([1, 2, 3, 4], dtype=np.int16)})})

        group = zarr.open_group(store, mode="r")
        output_array = group["particles"]["x"]  # type: ignore[invalid-argument-type]
        assert isinstance(output_array, zarr.Array)
        assert output_array.dtype == np.dtype(np.int16)

    def test_static_output_uses_particle_property_dtype_when_dtype_is_none(
        self, zarr_store, make_output_builder, make_particles_view
    ) -> None:
        store = zarr_store
        builder = make_output_builder(store)
        builder.add_static_output("particles", "density", Output("temperature", dtype=None))

        builder.build({
            "particles": make_particles_view(4, {"temperature": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)})
        })

        group = zarr.open_group(store, mode="r")
        output_array = group["particles"]["density"]  # type: ignore[invalid-argument-type]
        assert isinstance(output_array, zarr.Array)
        assert output_array.dtype == np.dtype(np.float32)

    def test_explicit_dtype_overrides_particle_property_dtype(
        self, zarr_store, make_output_builder, make_particles_view
    ) -> None:
        store = zarr_store
        builder = make_output_builder(store)
        builder.add_output("particles", "x", Output("mass", dtype=np.float32))

        builder.build({"particles": make_particles_view(4, {"mass": np.array([1, 2, 3, 4], dtype=np.int16)})})

        group = zarr.open_group(store, mode="r")
        output_array = group["particles"]["x"]  # type: ignore[invalid-argument-type]
        assert isinstance(output_array, zarr.Array)
        assert output_array.dtype == np.dtype(np.float32)
