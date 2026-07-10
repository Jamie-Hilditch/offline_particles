"""Tests for simulation builder ordering and output-builder inputs."""

from __future__ import annotations

import numpy as np

from offline_particles.fieldset import Fieldset
from offline_particles.kernels import ParticlePropertyDeclaration
from offline_particles.output import AbstractOutputWriter, AbstractOutputWriterBuilder
from offline_particles.particles import ParticlesView
from offline_particles.simulation import ParticleSet, SimulationBuilder
from offline_particles.timestepping import Timestepper


class _NoOpTimestepper(Timestepper):
    def run_step(self, particles, launcher, clock) -> None:
        pass


class _RecordingWriter(AbstractOutputWriter):
    @property
    def name(self) -> str:
        return "recording"

    @property
    def outputs(self):
        return ()

    @property
    def static_outputs(self):
        return ()

    def write_time(self, state) -> None:
        pass

    def write_output(self, particle_set: str, name: str, state) -> None:
        pass

    def finalise_write_round(self, state) -> None:
        pass

    def write_static_output(self, particle_set: str, name: str, state) -> None:
        pass


class _RecordingBuilder(AbstractOutputWriterBuilder):
    def __init__(self) -> None:
        self.received_particles: ParticlesView | None = None
        self.received_dtype: np.dtype | None = None

    @property
    def name(self) -> str:
        return "recording"

    @property
    def outputs(self):
        return ()

    @property
    def static_outputs(self):
        return ()

    def add_output(self, particle_set: str, name: str, output, **kwargs) -> None:
        pass

    def remove_output(self, particle_set: str, name: str) -> None:
        pass

    def add_static_output(self, particle_set: str, name: str, output, **kwargs) -> None:
        pass

    def remove_static_output(self, particle_set: str, name: str) -> None:
        pass

    def build(self, particles: dict[str, ParticlesView], time_type):
        self.received_particles = particles["pset"]
        self.received_dtype = particles["pset"]["mass"].dtype
        return _RecordingWriter()


def test_build_simulation_passes_particles_view_to_output_builder(make_clock, make_bound_noop_kernel) -> None:
    fieldset = Fieldset(1, 1, 1, 1)
    timestepper = _NoOpTimestepper()
    mass_kernel = make_bound_noop_kernel(particle_properties=[ParticlePropertyDeclaration("mass", np.float32)])
    timestepper.add_pre_step_kernels(mass_kernel)
    particle_set = ParticleSet("pset", 3, timestepper, include_validation_kernel=False)
    clock = make_clock(np.array([0.0, 1.0], dtype=np.float64), 1.0)
    builder = SimulationBuilder(clock, fieldset, particle_set)

    recording_builder = _RecordingBuilder()
    builder.add_output_writer(recording_builder, n=1)

    builder.build_simulation()

    assert isinstance(recording_builder.received_particles, ParticlesView)
    assert recording_builder.received_dtype == np.dtype(np.float32)
