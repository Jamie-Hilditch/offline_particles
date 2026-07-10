import numpy as np
import pytest

from offline_particles.kernels import BoundKernel, ParticleKernel
from offline_particles.timestepping import Clock


@pytest.fixture(autouse=True)
def add_doctest_namespace(doctest_namespace):
    doctest_namespace["np"] = np


@pytest.fixture
def make_clock():
    def _make_clock(time_array: np.ndarray, dt: float) -> Clock:
        return Clock(time_array, np.float64(dt))

    return _make_clock


@pytest.fixture
def make_bound_noop_kernel():
    def _make_bound_noop_kernel(particle_properties=None) -> BoundKernel:
        return BoundKernel(ParticleKernel(lambda pp, sc, fd: None, particle_properties or []))

    return _make_bound_noop_kernel
