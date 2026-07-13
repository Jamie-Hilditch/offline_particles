"""Tests for timed_activation kernels (timed release and retirement)."""

import numpy as np
import pytest

from offline_particles.kernels.status import Status
from offline_particles.kernels.timed_activation import (
    construct_activate_released_particles_kernel,
    construct_deactivate_retired_particles_kernel,
)


class TestActivateReleasedParticlesKernel:
    @pytest.mark.parametrize("dt", [1.0, -1.0])
    def test_releases_particles_at_or_past_release_time_to_initialising(self, dt: float) -> None:
        kernel = construct_activate_released_particles_kernel("release_time")

        status = np.array(
            [
                np.uint8(Status.PRE_RELEASE),
                np.uint8(Status.PRE_RELEASE),
                np.uint8(Status.PRE_RELEASE),
                np.uint8(Status.NORMAL),
            ],
            dtype=np.uint8,
        )
        release_time = np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float64)

        kernel.kernel(
            {"status": status, "release_time": release_time},
            {"_time": np.float64(1.0), "_dt": np.float64(dt)},
            {},
        )

        if dt > 0:
            # forward in time: time >= release_time releases particles 0 and 1, not particle 2
            expected = [Status.INITIALISING, Status.INITIALISING, Status.PRE_RELEASE, Status.NORMAL]
        else:
            # backward in time: time <= release_time releases particles 1 and 2, not particle 0
            expected = [Status.PRE_RELEASE, Status.INITIALISING, Status.INITIALISING, Status.NORMAL]

        np.testing.assert_array_equal(status, np.array([np.uint8(s) for s in expected], dtype=np.uint8))

    def test_only_affects_pre_release_particles(self) -> None:
        kernel = construct_activate_released_particles_kernel("release_time")

        status = np.array(
            [np.uint8(Status.NORMAL), np.uint8(Status.INACTIVE), np.uint8(Status.POST_RETIREMENT)],
            dtype=np.uint8,
        )
        release_time = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        kernel.kernel(
            {"status": status, "release_time": release_time},
            {"_time": np.float64(5.0), "_dt": np.float64(1.0)},
            {},
        )

        np.testing.assert_array_equal(
            status,
            np.array(
                [np.uint8(Status.NORMAL), np.uint8(Status.INACTIVE), np.uint8(Status.POST_RETIREMENT)],
                dtype=np.uint8,
            ),
        )

    def test_leaves_pre_release_particles_untouched_before_release_time(self) -> None:
        kernel = construct_activate_released_particles_kernel("release_time")

        status = np.array([np.uint8(Status.PRE_RELEASE)], dtype=np.uint8)
        release_time = np.array([10.0], dtype=np.float64)

        kernel.kernel(
            {"status": status, "release_time": release_time},
            {"_time": np.float64(1.0), "_dt": np.float64(1.0)},
            {},
        )

        np.testing.assert_array_equal(status, np.array([np.uint8(Status.PRE_RELEASE)], dtype=np.uint8))


class TestDeactivateRetiredParticlesKernel:
    def test_retires_active_particles_at_or_past_retirement_time(self) -> None:
        kernel = construct_deactivate_retired_particles_kernel("retirement_time")

        status = np.array([np.uint8(Status.NORMAL), np.uint8(Status.PRE_RELEASE)], dtype=np.uint8)
        retirement_time = np.array([0.0, 0.0], dtype=np.float64)

        kernel.kernel(
            {"status": status, "retirement_time": retirement_time},
            {"_time": np.float64(1.0), "_dt": np.float64(1.0)},
            {},
        )

        # PRE_RELEASE is already inactive, so it's skipped and unaffected by retirement
        np.testing.assert_array_equal(
            status, np.array([np.uint8(Status.POST_RETIREMENT), np.uint8(Status.PRE_RELEASE)], dtype=np.uint8)
        )
