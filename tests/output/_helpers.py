"""Helpers for output tests."""

from __future__ import annotations

import numpy as np

from offline_particles.particles import Particles, ParticlesView


def make_particles_view(nparticles: int, property_values: dict[str, np.ndarray] | None = None) -> ParticlesView:
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
