"""Shared test support for tests/output."""

import numpy as np
import pytest
import zarr.storage

from offline_particles.output import ZarrOutputBuilder
from offline_particles.particles import Particles, ParticlesView


@pytest.fixture
def zarr_store() -> zarr.storage.StoreLike:
    return zarr.storage.MemoryStore()


@pytest.fixture
def make_output_builder():
    def _make_output_builder(store: zarr.storage.StoreLike) -> ZarrOutputBuilder:
        return ZarrOutputBuilder("test_writer", store)

    return _make_output_builder


@pytest.fixture
def make_particles_view():
    def _make_particles_view(nparticles: int, property_values: dict[str, np.ndarray] | None = None) -> ParticlesView:
        """Create a ``ParticlesView`` for testing output builders.

        Parameters
        ----------
        nparticles : int
            The number of particles to allocate.
        property_values : dict[str, np.ndarray] | None, optional
            Property arrays to seed into the particle set.

        Returns
        -------
        ParticlesView
            A read-only view of the constructed particles.
        """
        property_dtypes: dict[str, np.dtype] = {"xidx": np.dtype(np.float64), "yidx": np.dtype(np.float64)}
        if property_values:
            for name, values in property_values.items():
                property_dtypes[name] = np.asarray(values).dtype

        particles = Particles(nparticles, property_dtypes)

        if property_values:
            for name, values in property_values.items():
                particles[name][:] = np.asarray(values, dtype=particles[name].dtype)

        return ParticlesView(particles)

    return _make_particles_view
