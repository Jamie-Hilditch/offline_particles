"""Submodule defining the top-level particle simulation class."""

import numpy as np

from .fieldset import Fieldset
from .launcher import Launcher
from .particle_kernel import merge_particle_fields
from .tasks import SimulationState, Task
from .timesteppers import Timestepper


class ParticleSimulation: 
    """Class representing a particle simulation."""

    def __init__(self, nparticles: int, timestepper: Timestepper, fieldset: Fieldset, tasks: dict[str, Task]) -> None:
        """Initialize the ParticleSimulation.

        Args:
            builder: The SimulationBuilder used to configure the simulation.
        """
        self._timestepper = timestepper
        self._fieldset = fieldset
        self._tasks = tasks

        # create launcher and register kernel data functions
        self._launcher = Launcher(fieldset)
        self._launcher.maybe_increase_index_padding(timestepper.index_padding)
        self._launcher.register_scalar_data_sources_from_object(self._timestepper)
        for task in self._tasks.values():
            self._launcher.register_scalar_data_sources_from_object(task)

        # construct the particle dtype 
        kernels = list(self._timestepper.kernels())
        for task in self._tasks.values():
            kernels.append(task.kernels())
        particle_fields = merge_particle_fields(kernels)
        self._particle_dtype = np.dtype([(field, dtype) for field, dtype in particle_fields.items()])

        # create particles array
        self._particles = np.empty((nparticles,), dtype=self._particle_dtype)

    @property
    def timestepper(self) -> Timestepper:
        """Get the timestepper used in the simulation.

        Returns:
            Timestepper: The timestepper instance.
        """
        return self._timestepper

    @property
    def fieldset(self) -> Fieldset:
        """Get the fieldset used in the simulation.

        Returns:
            Fieldset: The fieldset instance.
        """
        return self._fieldset

    @property
    def time(self) -> float:
        """Get the current simulation time.

        Returns:
            float: The current time of the simulation.
        """
        return self._timestepper.time

    @property
    def dt(self) -> float:
        """Get the timestep size.

        Returns:
            float: The size of each timestep in the simulation.
        """
        return self._timestepper.dt

    @property
    def tidx(self) -> float:
        """Get the current timestep index.

        Returns:
            float: The index of the current timestep.
        """
        return self._timestepper.tidx

    @property
    def particles(self):
        """Get the current particle data.

        Returns:
            The current state of the particles in the simulation.
        """
        return self._particles

    @property
    def state(self) -> SimulationState:
        """Get the current simulation state.

        Returns:
            SimulationState: A named tuple containing time, dt, tidx, and particles.
        """
        return SimulationState(
            time=self.time,
            dt=self.dt,
            tidx=self.tidx,
            particles=self.particles,
        )

    def step(self) -> None:
        """Advance the particle simulation by one timestep."""
        self._timestepper.timestep_particles(self._particles, self._launcher)


class SimulationBuilder:
    def __init__(self, timestepper: Timestepper, fieldset: Fieldset, **tasks: Task) -> None:
        """Class for build a ParticleSimulation.

        Args:
            timestepper: The launcher responsible for executing kernels.
        """
        self._timestepper = timestepper
        self._fieldset = fieldset

        # tasks
        self._tasks = {}
        for key, task in tasks.items():
            self.add_task(key, task)

    def add_task(self, key: str, task: Task) -> None:
        """Add a task to the simulation.

        Args:
            task: The task to be added to the launcher.
        """
        if key in self._tasks:
            raise KeyError(f"Task with key '{key}' already exists in the simulation. Please remove it before adding a new one.")
        self._tasks[key] = task

    def remove_task(self, key: str) -> None:
        """Remove a task from the simulation.

        Args:
            key: The key of the task to be removed.
        """
        if key not in self._tasks:
            raise KeyError(f"Task with key '{key}' does not exist in the simulation. Cannot remove.")
        del self._tasks[key]

    def build_simulation(self, nparticles: int) -> ParticleSimulation:
        """Build and return the ParticleSimulation.

        Args:
            nparticles: The number of particles in the simulation.
        """
        return ParticleSimulation(
            nparticles=nparticles,
            timestepper=self._timestepper,
            fieldset=self._fieldset,
            tasks=self._tasks,
        )