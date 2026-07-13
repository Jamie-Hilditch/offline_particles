"""Tests for validation kernel wiring into timesteppers and simulations."""

import numpy as np

from offline_particles.fieldset import Fieldset
from offline_particles.kernels import BoundKernel
from offline_particles.kernels.status import Status
from offline_particles.launcher import Launcher
from offline_particles.particles import Particles
from offline_particles.simulation import ParticleSet, SimulationBuilder
from offline_particles.timestepping import Timestepper


class _RecordingLauncher(Launcher):
    def __init__(self) -> None:
        super().__init__(Fieldset(1, 1, 1, 1), history_size=1)
        self.calls = []

    def launch_kernel(self, bound_kernel: BoundKernel, particles: Particles, tinfo) -> None:
        self.calls.append((bound_kernel, particles, tinfo))


class _RecordingLifecycleTimestepper(Timestepper):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def run_validation(self, particles, launcher, clock) -> None:
        self.calls.append("validation")

    def run_pre_step(self, particles, launcher, clock) -> None:
        self.calls.append("pre_step")

    def run_step(self, particles, launcher, clock) -> None:
        self.calls.append("step")

    def run_post_step(self, particles, launcher, clock) -> None:
        self.calls.append("post_step")


def _make_builder(
    timestepper: Timestepper,
    make_clock,
    *,
    include_validation_kernel: bool = True,
    fieldset: Fieldset | None = None,
) -> SimulationBuilder:
    fieldset = fieldset or Fieldset(2, 3, 4, 5)
    particle_set = ParticleSet(
        "pset",
        3,
        timestepper,
        include_validation_kernel=include_validation_kernel,
    )
    return SimulationBuilder(make_clock(np.array([0.0, 1.0, 2.0], dtype=np.float64), 1.0), fieldset, particle_set)


class TestTimestepperValidationKernelStorage:
    def test_add_validation_kernels_appends_in_order(self, make_bound_noop_kernel, noop_timestepper) -> None:
        timestepper = noop_timestepper
        kernel_0 = make_bound_noop_kernel()
        kernel_1 = make_bound_noop_kernel()

        timestepper.add_validation_kernels(kernel_0, kernel_1)

        assert timestepper.validation_kernels == [kernel_0, kernel_1]

    def test_kernels_iterator_places_validation_before_pre_step(self, make_bound_noop_kernel, noop_timestepper) -> None:
        timestepper = noop_timestepper
        validation_kernel = make_bound_noop_kernel()
        pre_step_kernel = make_bound_noop_kernel()

        timestepper.add_validation_kernels(validation_kernel)
        timestepper.add_pre_step_kernels(pre_step_kernel)

        assert list(timestepper.kernels) == [timestepper._initialise_status_kernel, validation_kernel, pre_step_kernel]

    def test_run_validation_launches_each_validation_kernel(
        self, make_clock, make_bound_noop_kernel, noop_timestepper
    ) -> None:
        timestepper = noop_timestepper
        kernel_0 = make_bound_noop_kernel()
        kernel_1 = make_bound_noop_kernel()
        timestepper.add_validation_kernels(kernel_0, kernel_1)

        launcher = _RecordingLauncher()
        particles = Particles(1, {})
        clock = make_clock(np.array([0.0, 1.0], dtype=np.float64), 1.0)

        timestepper.run_validation(particles, launcher, clock)

        assert [call[0] for call in launcher.calls] == [kernel_0, kernel_1]
        assert all(call[1] is particles for call in launcher.calls)
        assert all(call[2] == clock.tinfo for call in launcher.calls)


class TestSetInitialStatus:
    def test_set_initial_status_replaces_finalize_kernel(self, noop_timestepper) -> None:
        timestepper = noop_timestepper
        default_kernel = timestepper._initialise_status_kernel

        timestepper.set_initial_status(Status.MULTISTEP_1)

        assert timestepper._initialise_status_kernel is not default_kernel

        status = np.array([np.uint8(Status.INITIALISING)], dtype=np.uint8)
        timestepper._initialise_status_kernel.kernel({"status": status}, {}, {})
        assert status[0] == np.uint8(Status.MULTISTEP_1)


class TestSimulationBuilderValidationInjection:
    def test_build_simulation_injects_validation_kernel_from_fieldset_bbox(self, make_clock, noop_timestepper) -> None:
        fieldset = Fieldset(
            2, 3, 4, 5, zidx_min=2.0, zidx_max=4.0, yidx_min=10.0, yidx_max=20.0, xidx_min=30.0, xidx_max=40.0
        )
        timestepper = noop_timestepper
        builder = _make_builder(timestepper, make_clock, fieldset=fieldset)

        builder.build_simulation()

        assert len(timestepper.validation_kernels) == 1

        injected_kernel = timestepper.validation_kernels[0]
        status = np.array([np.uint8(Status.NORMAL), np.uint8(Status.NORMAL)], dtype=np.uint8)
        zidx = np.array([3.0, 1.0], dtype=np.float64)
        yidx = np.array([15.0, 15.0], dtype=np.float64)
        xidx = np.array([35.0, 35.0], dtype=np.float64)

        injected_kernel.kernel(
            {
                "status": status,
                "zidx": zidx,
                "yidx": yidx,
                "xidx": xidx,
            },
            {},
            {},
        )

        np.testing.assert_array_equal(
            status,
            np.array([np.uint8(Status.NORMAL), np.uint8(Status.BELOW_BOTTOM)], dtype=np.uint8),
        )

    def test_build_simulation_can_skip_validation_kernel(self, make_clock, noop_timestepper) -> None:
        timestepper = noop_timestepper
        builder = _make_builder(timestepper, make_clock, include_validation_kernel=False)

        builder.build_simulation()

        assert timestepper.validation_kernels == []


class TestSimulationValidationOrder:
    def test_step_runs_validation_before_pre_step(self, make_clock) -> None:
        timestepper = _RecordingLifecycleTimestepper()
        builder = _make_builder(timestepper, make_clock, include_validation_kernel=False)
        sim = builder.build_simulation()

        sim.step()

        assert timestepper.calls == ["validation", "pre_step", "step", "post_step"]

    def test_step_runs_initialisation_kernel_every_step(self, make_clock, noop_timestepper) -> None:
        timestepper = noop_timestepper
        builder = _make_builder(timestepper, make_clock, include_validation_kernel=False)
        sim = builder.build_simulation()
        particles = sim._particles["pset"]

        # particles default to INITIALISING; the first step() call finalizes them to NORMAL
        sim.step()
        np.testing.assert_array_equal(particles["status"], np.full(3, np.uint8(Status.NORMAL)))

        # a particle transitioning back to INITIALISING mid-simulation (e.g. via timed release)
        # is picked up and finalized by the *next* step() call too, not just once at sim start
        particles["status"][1] = np.uint8(Status.INITIALISING)
        sim.step()
        np.testing.assert_array_equal(particles["status"], np.full(3, np.uint8(Status.NORMAL)))
