import numpy as np
import pytest

from offline_particles.timestepping import Clock


@pytest.fixture(autouse=True)
def add_doctest_namespace(doctest_namespace):
    doctest_namespace["np"] = np


@pytest.fixture
def make_clock():
    def _make_clock(time_array: np.ndarray, dt: float) -> Clock:
        return Clock(time_array, np.float64(dt))

    return _make_clock
