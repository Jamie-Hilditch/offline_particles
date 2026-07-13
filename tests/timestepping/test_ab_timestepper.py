"""Tests for the ABTimestepper class."""

import numpy as np
import pytest

from offline_particles.fieldset import Fieldset
from offline_particles.kernels.status import Status
from offline_particles.launcher import Launcher
from offline_particles.particles import Particles
from offline_particles.timestepping import ABTimestepper


def _make_particles() -> Particles:
    particles = Particles(
        4,
        {
            "status": np.dtype(np.uint8),
            "zidx": np.dtype(np.float64),
            "yidx": np.dtype(np.float64),
            "xidx": np.dtype(np.float64),
        },
    )
    particles["status"][:] = np.array(
        [
            np.uint8(Status.NORMAL),
            np.uint8(Status.INACTIVE),
            np.uint8(Status.NONFINITE),
            np.uint8(Status.INITIALISING),
        ],
        dtype=np.uint8,
    )
    return particles


class _RecordingLauncher:
    def __init__(self) -> None:
        self.calls = []

    def launch_kernel(self, bound_kernel, particles, tinfo) -> None:
        self.calls.append((bound_kernel, particles, tinfo))


class TestABTimestepperConstruction:
    def test_rejects_unsupported_orders(self) -> None:
        with pytest.raises(ValueError, match="Only orders 2 and 3"):
            ABTimestepper(order=1)

    @pytest.mark.parametrize("order", [2, 3])
    def test_adds_initialisation_kernel(self, order: int) -> None:
        timestepper = ABTimestepper(order=order)

        assert len(timestepper.initialisation_kernels) == 0
        assert timestepper._initialise_status_kernel is not None

    def test_stores_index_padding(self) -> None:
        timestepper = ABTimestepper(order=2, index_padding=3)

        assert timestepper.index_padding == 3


class TestABTimestepperPrognosticKernelConstruction:
    def test_ab2_default_derivative_bindings(self) -> None:
        timestepper = ABTimestepper(order=2)

        timestepper.add_prognostic_property_kernel("x")

        kernel = timestepper._ab_update_kernels[0]
        assert kernel.particle_property_bindings["prop"] == "x"
        assert kernel.particle_property_bindings["dprop_0"] == "x_d0"
        assert kernel.particle_property_bindings["dprop_1"] == "x_d1"
        assert "dprop_2" not in kernel.particle_property_bindings

    def test_ab3_custom_derivative_bindings(self) -> None:
        timestepper = ABTimestepper(order=3)

        timestepper.add_prognostic_property_kernel("x", dprop_0="dx0", dprop_1="dx1", dprop_2="dx2")

        kernel = timestepper._ab_update_kernels[0]
        assert kernel.particle_property_bindings["prop"] == "x"
        assert kernel.particle_property_bindings["dprop_0"] == "dx0"
        assert kernel.particle_property_bindings["dprop_1"] == "dx1"
        assert kernel.particle_property_bindings["dprop_2"] == "dx2"


class TestABTimestepperExecution:
    def test_run_step_launches_tendency_then_update_then_status_bump(self, make_clock, make_bound_noop_kernel) -> None:
        timestepper = ABTimestepper(order=2)
        tendency_0 = make_bound_noop_kernel()
        tendency_1 = make_bound_noop_kernel()
        ab_update = make_bound_noop_kernel()

        timestepper.add_tendency_kernels(tendency_0, tendency_1)
        timestepper.add_ab_update_kernels(ab_update)

        launcher = _RecordingLauncher()
        particles = _make_particles()
        clock = make_clock(np.array([0.0, 1.0, 2.0], dtype=np.float64), 1.0)

        timestepper.run_step(particles, launcher, clock)  # type: ignore

        launched_kernels = [call[0] for call in launcher.calls]
        assert launched_kernels[:-1] == [tendency_0, tendency_1, ab_update]
        assert launched_kernels[-1] is timestepper._bump_status_kernel

        for _, launched_particles, launched_tinfo in launcher.calls:
            assert launched_particles is particles
            assert launched_tinfo == clock.tinfo

    @pytest.mark.parametrize(
        ("order", "expected_status"),
        [
            (2, np.uint8(Status.MULTISTEP_1)),
            (3, np.uint8(Status.MULTISTEP_2)),
        ],
    )
    def test_run_initialisation_sets_expected_ab_statuses(
        self, order: int, expected_status: np.uint8, make_clock
    ) -> None:
        timestepper = ABTimestepper(order=order)
        particles = _make_particles()
        launcher = Launcher(Fieldset(1, 1, 1, 1), history_size=1)
        clock = make_clock(np.array([0.0, 1.0, 2.0], dtype=np.float64), 1.0)

        timestepper.run_initialisation(particles, launcher, clock)

        np.testing.assert_array_equal(
            particles["status"],
            np.array(
                [
                    np.uint8(Status.NORMAL),
                    np.uint8(Status.INACTIVE),
                    np.uint8(Status.NONFINITE),
                    expected_status,
                ],
                dtype=np.uint8,
            ),
        )
