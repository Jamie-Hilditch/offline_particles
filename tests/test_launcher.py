"""Tests for launcher.py's particle bounding-box computation."""

import numpy as np

from offline_particles.fields import StaticField
from offline_particles.fieldset import Fieldset
from offline_particles.kernels import BoundKernel, FieldDataDeclaration, ParticleKernel
from offline_particles.kernels.status import Status
from offline_particles.launcher import Launcher, _compute_particle_bounds
from offline_particles.particles import Particles
from offline_particles.spatial_arrays import BBox


def test_compute_particle_bounds_includes_initialising_particles() -> None:
    status = np.array(
        [np.uint8(Status.NORMAL), np.uint8(Status.INITIALISING), np.uint8(Status.PRE_RELEASE)],
        dtype=np.uint8,
    )
    zidx = np.array([1.0, 100.0, -100.0], dtype=np.float64)
    yidx = np.array([1.0, 100.0, -100.0], dtype=np.float64)
    xidx = np.array([1.0, 100.0, -100.0], dtype=np.float64)

    zmin, zmax, ymin, ymax, xmin, xmax, any_active = _compute_particle_bounds(status, zidx, yidx, xidx)

    # bounds span the NORMAL and INITIALISING particles (despite INITIALISING carrying the
    # INACTIVE bit), excluding the PRE_RELEASE particle
    assert any_active
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

    zmin, zmax, ymin, ymax, xmin, xmax, any_active = _compute_particle_bounds(status, zidx, yidx, xidx)

    # only the NORMAL particle contributes
    assert any_active
    assert (zmin, zmax) == (1.0, 1.0)
    assert (ymin, ymax) == (1.0, 1.0)
    assert (xmin, xmax) == (1.0, 1.0)


def test_compute_particle_bounds_reports_no_active_particles_when_all_inactive() -> None:
    status = np.array(
        [np.uint8(Status.NONFINITE), np.uint8(Status.PRE_RELEASE), np.uint8(Status.POST_RETIREMENT)],
        dtype=np.uint8,
    )
    zidx = np.array([1.0, 100.0, -100.0], dtype=np.float64)
    yidx = np.array([1.0, 100.0, -100.0], dtype=np.float64)
    xidx = np.array([1.0, 100.0, -100.0], dtype=np.float64)

    zmin, zmax, ymin, ymax, xmin, xmax, any_active = _compute_particle_bounds(status, zidx, yidx, xidx)

    assert not any_active
    assert (zmin, zmax) == (np.inf, -np.inf)
    assert (ymin, ymax) == (np.inf, -np.inf)
    assert (xmin, xmax) == (np.inf, -np.inf)


def test_compute_particle_bounds_reports_no_active_particles_when_empty() -> None:
    status = np.array([], dtype=np.uint8)
    zidx = np.array([], dtype=np.float64)
    yidx = np.array([], dtype=np.float64)
    xidx = np.array([], dtype=np.float64)

    _, _, _, _, _, _, any_active = _compute_particle_bounds(status, zidx, yidx, xidx)

    assert not any_active


class TestConstructBBox:
    def test_returns_none_for_zero_particles(self) -> None:
        launcher = Launcher(Fieldset(1, 4, 4, 4), history_size=2)
        particles = Particles(0, {})

        assert launcher.construct_bbox(particles) is None

    def test_returns_none_when_all_particles_inactive(self) -> None:
        launcher = Launcher(Fieldset(1, 4, 4, 4), history_size=2)
        particles = Particles(3, {})
        particles["status"][:] = np.uint8(Status.NONFINITE)

        assert launcher.construct_bbox(particles) is None

    def test_returns_bbox_when_active_particles_present(self) -> None:
        launcher = Launcher(Fieldset(1, 4, 4, 4), history_size=1)
        particles = Particles(1, {})
        particles["status"][:] = np.uint8(Status.NORMAL)
        particles["zidx"][:] = 2.0
        particles["yidx"][:] = 2.0
        particles["xidx"][:] = 2.0

        bbox = launcher.construct_bbox(particles)

        assert bbox == BBox(zmin=2.0, zmax=2.0, ymin=2.0, ymax=2.0, xmin=2.0, xmax=2.0)

    def test_leaves_history_untouched_when_no_active_particles(self) -> None:
        launcher = Launcher(Fieldset(1, 4, 4, 4), history_size=2)
        active_particles = Particles(1, {})
        active_particles["status"][:] = np.uint8(Status.NORMAL)
        active_particles["zidx"][:] = 1.0
        active_particles["yidx"][:] = 1.0
        active_particles["xidx"][:] = 1.0
        launcher.construct_bbox(active_particles)

        inactive_particles = Particles(1, {})
        inactive_particles["status"][:] = np.uint8(Status.NONFINITE)
        assert launcher.construct_bbox(inactive_particles) is None

        # the degenerate call left the history exactly as the earlier active call set it
        assert list(launcher._zmin_history) == [1.0]
        assert list(launcher._zmax_history) == [1.0]


class TestLaunchKernelSkipsWhenNoActiveParticles:
    def test_skips_field_fetch_and_kernel_call(self, make_clock, monkeypatch) -> None:
        fieldset = Fieldset(1, 4, 4, 4)
        fieldset.add_field("h", StaticField.from_numpy(np.ones((4, 4)), axes=("Y", "X"), staggers=("center", "center")))
        launcher = Launcher(fieldset, history_size=1)

        kernel_called = False

        def kernel_fn(particle_properties, scalars, field_data) -> None:
            nonlocal kernel_called
            kernel_called = True

        bound_kernel = BoundKernel(ParticleKernel(kernel_fn, field_data=[FieldDataDeclaration("h", np.float64)]))

        def _fail_get_field_data(name, time_index, bbox):
            raise AssertionError("get_field_data should not be called when there are no active particles")

        monkeypatch.setattr(launcher, "get_field_data", _fail_get_field_data)

        particles = Particles(0, {})
        clock = make_clock(np.array([0.0, 1.0], dtype=np.float64), 1.0)

        # previously raised deep inside field-data subsetting; must now be a no-op
        launcher.launch_kernel(bound_kernel, particles, clock.tinfo)

        assert not kernel_called

    def test_runs_kernel_when_active_particles_present(self, make_clock, monkeypatch) -> None:
        fieldset = Fieldset(1, 4, 4, 4)
        fieldset.add_field("h", StaticField.from_numpy(np.ones((4, 4)), axes=("Y", "X"), staggers=("center", "center")))
        launcher = Launcher(fieldset, history_size=1)

        kernel_called = False

        def kernel_fn(particle_properties, scalars, field_data) -> None:
            nonlocal kernel_called
            kernel_called = True

        bound_kernel = BoundKernel(ParticleKernel(kernel_fn, field_data=[FieldDataDeclaration("h", np.float64)]))

        particles = Particles(1, {})
        particles["status"][:] = np.uint8(Status.NORMAL)
        particles["yidx"][:] = 1.0
        particles["xidx"][:] = 1.0
        clock = make_clock(np.array([0.0, 1.0], dtype=np.float64), 1.0)

        launcher.launch_kernel(bound_kernel, particles, clock.tinfo)

        assert kernel_called
