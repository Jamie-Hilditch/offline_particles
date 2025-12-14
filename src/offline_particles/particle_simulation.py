"""Submodule defining the top-level particle simulation class."""



from .fieldset import Fieldset
from .kernels import merge_particle_fields
from .launcher import Launcher
from .particles import Particles
from .tasks import SimulationState, Task
from .timesteppers import Timestepper


class ParticleSimulation:
    """Class representing a particle simulation."""

    def __init__(
        self,
        nparticles: int,
        timestepper: Timestepper,
        fieldset: Fieldset,
        tasks: dict[str, Task],
        *,
        zidx_bounds: tuple[float, float] | None = None,
        yidx_bounds: tuple[float, float] | None = None,
        xidx_bounds: tuple[float, float] | None = None,
    ) -> None:
        """Initialize the ParticleSimulation.

        Args:
            builder: The SimulationBuilder used to configure the simulation.
        """
        self._timestepper = timestepper
        self._fieldset = fieldset
        self._tasks = tasks

        # set default index bounds if not provided
        if zidx_bounds is None:
            zidx_bounds = (0.0, float(self._fieldset.z_size - 1))
        if yidx_bounds is None:
            yidx_bounds = (0.0, float(self._fieldset.y_size - 1))
        if xidx_bounds is None:
            xidx_bounds = (0.0, float(self._fieldset.x_size - 1))
        self._zidx_bounds = zidx_bounds
        self._yidx_bounds = yidx_bounds
        self._xidx_bounds = xidx_bounds

        # create launcher and register kernel data functions
        self._launcher = Launcher(fieldset)
        self._launcher.maybe_increase_index_padding(timestepper.index_padding)
        self._launcher.register_scalar_data_sources_from_object(self._timestepper)
        for task in self._tasks.values():
            self._launcher.register_scalar_data_sources_from_object(task)

        # register index bounds as scalar data sources
        self._launcher.register_scalar_data_source(
            "zidx_min", lambda tidx: self._zidx_bounds[0]
        )
        self._launcher.register_scalar_data_source(
            "zidx_max", lambda tidx: self._zidx_bounds[1]
        )
        self._launcher.register_scalar_data_source(
            "yidx_min", lambda tidx: self._yidx_bounds[0]
        )
        self._launcher.register_scalar_data_source(
            "yidx_max", lambda tidx: self._yidx_bounds[1]
        )
        self._launcher.register_scalar_data_source(
            "xidx_min", lambda tidx: self._xidx_bounds[0]
        )
        self._launcher.register_scalar_data_source(
            "xidx_max", lambda tidx: self._xidx_bounds[1]
        )

        # construct the particles
        kernels = list(self._timestepper.kernels())
        for task in self._tasks.values():
            kernels.append(task.kernels())
        # merge required particle fields from all kernels
        particle_fields = merge_particle_fields(kernels)
        self._particles = Particles(nparticles, **particle_fields)

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
    def __init__(
        self,
        timestepper: Timestepper,
        fieldset: Fieldset,
        *,
        zidx_bounds: tuple[float, float] | None = None,
        yidx_bounds: tuple[float, float] | None = None,
        xidx_bounds: tuple[float, float] | None = None,
        **tasks: Task,
    ) -> None:
        """Class for build a ParticleSimulation.

        Args:
            timestepper: The launcher responsible for executing kernels.
        """
        self._timestepper = timestepper
        self._fieldset = fieldset
        self._zidx_bounds = zidx_bounds
        self._yidx_bounds = yidx_bounds
        self._xidx_bounds = xidx_bounds

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
            raise KeyError(
                f"Task with key '{key}' already exists in the simulation. Please remove it before adding a new one."
            )
        self._tasks[key] = task

    def remove_task(self, key: str) -> None:
        """Remove a task from the simulation.

        Args:
            key: The key of the task to be removed.
        """
        if key not in self._tasks:
            raise KeyError(
                f"Task with key '{key}' does not exist in the simulation. Cannot remove."
            )
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
            zidx_bounds=self._zidx_bounds,
            yidx_bounds=self._yidx_bounds,
            xidx_bounds=self._xidx_bounds,
        )
