"""Submodule defining the top-level particle simulation class."""

import functools
import itertools
import types
from typing import Mapping

from .events import (
    AbstractSchedule,
    Event,
    IterationSchedule,
    IterationScheduler,
    SimulationState,
    TimeSchedule,
    TimeScheduler,
)
from .fieldset import Fieldset
from .kernels import merge_particle_fields
from .launcher import Launcher
from .output import AbstractOutputWriter, AbstractOutputWriterBuilder
from .particles import Particles, ParticlesView
from .timesteppers import Timestepper


class Simulation:
    """Class representing a particle simulation."""

    def __init__(
        self,
        nparticles: int,
        timestepper: Timestepper,
        fieldset: Fieldset,
        iteration_scheduler: IterationScheduler,
        time_scheduler: TimeScheduler,
        output_writers: Mapping[str, AbstractOutputWriter],
        *,
        zidx_bounds: tuple[float, float] | None = None,
        yidx_bounds: tuple[float, float] | None = None,
        xidx_bounds: tuple[float, float] | None = None,
    ) -> None:
        """Initialize the Simulation.

        Args:
            builder: The SimulationBuilder used to configure the simulation.
        """
        self._timestepper = timestepper
        self._fieldset = fieldset
        self._iteration_scheduler = iteration_scheduler
        self._time_scheduler = time_scheduler

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
        # first gather all kernels
        kernels = list(self._timestepper.kernels())
        for event in self._iteration_scheduler.events:
            kernels.extend(event.kernels())
        for event in self._time_scheduler.events:
            kernels.extend(event.kernels())
        # then merge required particle fields from all kernels
        particle_fields = merge_particle_fields(kernels)
        self._particles = Particles(nparticles, **particle_fields)
        self._particles_view = ParticlesView(self._particles)

        # invoke any events scheduled for the initial time/iteration
        self._invoke_events()

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
    def iteration(self) -> int:
        """Get the current simulation iteration.

        Returns:
            int: The current iteration of the simulation.
        """
        return self._timestepper.iteration

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
        """A view into the current particle data.

        Returns:
            The current state of the particles in the simulation.
        """
        return self._particles_view

    @property
    def iteration_scheduler(self) -> IterationScheduler:
        """Get the iteration scheduler used in the simulation.

        Returns:
            IterationScheduler: The iteration scheduler instance.
        """
        return self._iteration_scheduler

    @property
    def time_scheduler(self) -> TimeScheduler:
        """Get the time scheduler used in the simulation.

        Returns:
            TimeScheduler: The time scheduler instance.
        """
        return self._time_scheduler

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
            particles=self._particles_view,
        )

    def _invoke_events(self) -> None:
        """Invoke any scheduled events at the current time or iteration."""
        for event in itertools.chain(
            self._iteration_scheduler(self.iteration), self._time_scheduler(self.time)
        ):
            # launch kernels
            for kernel in event.kernels:
                self._launcher.launch_kernel(kernel, self._particles, self.tidx)
            # invoke event function
            event(self.state)

    def step(self) -> None:
        """Advance the particle simulation by one timestep."""
        self._timestepper.timestep_particles(self._particles, self._launcher)
        # run any scheduled events
        self._invoke_events()


class SimulationBuilder:
    def __init__(
        self,
        timestepper: Timestepper,
        fieldset: Fieldset,
        *,
        zidx_bounds: tuple[float, float] | None = None,
        yidx_bounds: tuple[float, float] | None = None,
        xidx_bounds: tuple[float, float] | None = None,
    ) -> None:
        """Class for building a Simulation.

        Args:
            timestepper: The launcher responsible for executing kernels.
        """
        self._timestepper = timestepper
        self._fieldset = fieldset
        self._zidx_bounds = zidx_bounds
        self._yidx_bounds = yidx_bounds
        self._xidx_bounds = xidx_bounds

        # events
        self._iteration_scheduler = IterationScheduler()
        self._time_scheduler = TimeScheduler()

        # output writers
        self._output_writers: dict[str, AbstractOutputWriter] = dict()

    @functools.singledispatchmethod
    def add_event(self, schedule: AbstractSchedule, event: Event) -> None:
        """Add an event to the simulation.

        Args:
            schedule: The schedule for the event.
            event: The event to be added to the launcher.
        """
        raise NotImplementedError(
            "add_event only supports IterationSchedule or TimeSchedule"
        )

    @add_event.register
    def _(
        self,
        schedule: IterationSchedule,
        event: Event,
    ) -> None:
        """Add an iteration-based event to the simulation."""
        self._iteration_scheduler.register_event(schedule, event)

    @add_event.register
    def _(
        self,
        schedule: TimeSchedule,
        event: Event,
    ) -> None:
        """Add a time-based event to the simulation."""
        self._time_scheduler.register_event(schedule, event)

    def add_output_writer(self, builder: AbstractOutputWriterBuilder) -> None:
        """Add an output writer to the simulation.

        Args:
            writer: The output writer instance.
        """
        name = builder.name
        if name in self._output_writers:
            raise ValueError(f"Output writer '{name}' already exists.")
        self._output_writers[name] = builder

    def build_simulation(self, nparticles: int) -> Simulation:
        """Build and return the Simulation.

        Args:
            nparticles: The number of particles in the simulation.
        """
        # build output writers and make mapping immutable
        output_writers = {
            name: builder.build(nparticles)
            for name, builder in self._output_writers.items()
        }
        output_writers = types.MappingProxyType(output_writers)

        # add output events to the simulation
        for writer in output_writers.values():
            events = writer.create_events()
            schedule = writer.schedule
            for event in events:
                self.add_event(schedule, event)

        return Simulation(
            nparticles=nparticles,
            timestepper=self._timestepper,
            fieldset=self._fieldset,
            iteration_scheduler=self._iteration_scheduler,
            time_scheduler=self._time_scheduler,
            output_writers=output_writers,
            zidx_bounds=self._zidx_bounds,
            yidx_bounds=self._yidx_bounds,
            xidx_bounds=self._xidx_bounds,
        )
