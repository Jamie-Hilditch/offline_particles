"""Tests for launcher.py's particle bounding-box computation."""

import numpy as np

from offline_particles.kernels.status import Status
from offline_particles.launcher import _compute_particle_bounds


def test_compute_particle_bounds_includes_initialising_particles() -> None:
    status = np.array(
        [np.uint8(Status.NORMAL), np.uint8(Status.INITIALISING), np.uint8(Status.PRE_RELEASE)],
        dtype=np.uint8,
    )
    zidx = np.array([1.0, 100.0, -100.0], dtype=np.float64)
    yidx = np.array([1.0, 100.0, -100.0], dtype=np.float64)
    xidx = np.array([1.0, 100.0, -100.0], dtype=np.float64)

    zmin, zmax, ymin, ymax, xmin, xmax = _compute_particle_bounds(status, zidx, yidx, xidx)

    # bounds span the NORMAL and INITIALISING particles (despite INITIALISING carrying the
    # INACTIVE bit), excluding the PRE_RELEASE particle
    assert (zmin, zmax) == (1.0, 100.0)
    assert (ymin, ymax) == (1.0, 100.0)
    assert (xmin, xmax) == (1.0, 100.0)


def test_compute_particle_bounds_excludes_error_and_release_retirement_statuses() -> None:
    status = np.array(
        [
            np.uint8(Status.NORMAL),
            np.uint8(Status.NONFINITE),
            np.uint8(Status.PRE_RELEASE),
            np.uint8(Status.POST_RETIREMENT),
        ],
        dtype=np.uint8,
    )
    zidx = np.array([1.0, 100.0, -100.0, -200.0], dtype=np.float64)
    yidx = np.array([1.0, 100.0, -100.0, -200.0], dtype=np.float64)
    xidx = np.array([1.0, 100.0, -100.0, -200.0], dtype=np.float64)

    zmin, zmax, ymin, ymax, xmin, xmax = _compute_particle_bounds(status, zidx, yidx, xidx)

    # only the NORMAL particle contributes
    assert (zmin, zmax) == (1.0, 1.0)
    assert (ymin, ymax) == (1.0, 1.0)
    assert (xmin, xmax) == (1.0, 1.0)
