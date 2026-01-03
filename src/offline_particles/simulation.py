"""Submodule defining the top-level particle simulation class."""

import itertools
import types
from typing import Mapping, overload

import numpy as np
import numpy.typing as npt

from .events import (
    Event,
    IterationScheduler,
    SimulationState,
    TimeScheduler,
)
from .fieldset import Fieldset
from .kernels import merge_particle_fields
from .launcher import Launcher
from .output import AbstractOutputWriter, AbstractOutputWriterBuilder
from .particles import Particles, ParticlesView
from .timesteppers import Timestepper

type T = np.float64 | np.datetime64
type D = np.float64 | np.timedelta64


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
    ) -> None:
        """Initialize the Simulation.

        Args:
            builder: The SimulationBuilder used to configure the simulation.
        """
        self._timestepper = timestepper
        self._fieldset = fieldset
        self._iteration_scheduler = iteration_scheduler
        self._time_scheduler = time_scheduler

        # create launcher and register kernel data functions
        self._launcher = Launcher(fieldset)
        self._launcher.maybe_increase_index_padding(timestepper.index_padding)
        self._launcher.register_scalar_data_sources_from_object(self._timestepper)
        for event in self._iteration_scheduler.events:
            self._launcher.register_scalar_data_sources_from_object(event)
        for event in self._time_scheduler.events:
            self._launcher.register_scalar_data_sources_from_object(event)

        # construct the particles
        # first gather all kernels
        kernels = list(self._timestepper.kernels)
        for event in self._iteration_scheduler.events:
            kernels.extend(event.kernels)
        for event in self._time_scheduler.events:
            kernels.extend(event.kernels)
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
    def time(self) -> np.float64 | np.datetime64:
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
    def dt(self) -> np.float64 | np.timedelta64:
        """Get the timestep size.

        Returns:
            float: The size of each timestep in the simulation.
        """
        return self._timestepper.dt

    @property
    def tidx(self) -> np.float64:
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
            iteration=self.iteration,
            particles=self._particles_view,
        )

    def _invoke_events(self) -> None:
        """Invoke any scheduled events at the current time or iteration."""
        for event in itertools.chain(self._iteration_scheduler(self.iteration), self._time_scheduler(self.time)):
            # launch kernels
            for kernel in event.kernels:
                self._launcher.launch_kernel(kernel, self._particles, self.tidx)
            # invoke event function
            event(self.state)

    def set_indices(
        self,
        zidx: npt.ArrayLike | None = None,
        yidx: npt.ArrayLike | None = None,
        xidx: npt.ArrayLike | None = None,
    ) -> None:
        """Set the particles indices."""

        # first make the inputs compatible with the particle arrays
        # allow this to error if the shapes / types are incompatible
        # before modifying any particle data
        if zidx is not None:
            zidx = np.asarray(zidx, dtype=np.float64)
            zidx = np.broadcast_to(zidx, self._particles.zidx.shape)

        if yidx is not None:
            yidx = np.asarray(yidx, dtype=np.float64)
            yidx = np.broadcast_to(yidx, self._particles.yidx.shape)

        if xidx is not None:
            xidx = np.asarray(xidx, dtype=np.float64)
            xidx = np.broadcast_to(xidx, self._particles.xidx.shape)

        # now set the indices
        if zidx is not None:
            self._particles.zidx[:] = zidx
        if yidx is not None:
            self._particles.yidx[:] = yidx
        if xidx is not None:
            self._particles.xidx[:] = xidx

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
    ) -> None:
        """Class for building a Simulation.

        Args:
            timestepper: The launcher responsible for executing kernels.
        """
        self._timestepper = timestepper
        self._fieldset = fieldset

        # events
        self._iteration_scheduler = IterationScheduler()
        self._time_scheduler = TimeScheduler()

        # output writers
        self._output_writers: dict[str, tuple[AbstractOutputWriterBuilder, dict[str, ...]]] = dict()

    def every_n(self, n: int, event: Event, *, first: int | None = None) -> None:
        """Add an event that triggers every n iterations.

        Args:
            n (int): The interval in iterations between event triggers.
            event (Event): The event to be added.
            first (int, optional): The first iteration to trigger the event. Defaults to 0.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if first is None:
            first = 0
        self._iteration_scheduler.register_event(first, n, event)

    def every_dt(self, dt: D, event: Event, *, first: T | None = None) -> None:
        """Add an event that triggers every dt time units.

        Args:
            dt (D): The interval in time between event triggers.
            event (Event): The event to be added.
            first (T): The first time to trigger the event (defaults to timestepper.time).
        """
        # set default first time
        timestepper_time = self._timestepper.time
        if first is None:
            first = timestepper_time

        # check times are compatible
        try:
            _ = timestepper_time + dt  # type: ignore
        except TypeError as e:
            raise TypeError(
                f"Incompatible dt type {type(dt)} for timestepper time type {type(timestepper_time)}"
            ) from e
        try:
            _ = first + dt  # type: ignore
        except TypeError as e:
            raise TypeError(
                f"Incompatible first type {type(first)} for timestepper time type {type(timestepper_time)}"
            ) from e

        self._time_scheduler.register_event(first, dt, event)

    @overload
    def add_event(self, event: Event, *, n: int, first: int | None) -> None: ...

    @overload
    def add_event(self, event: Event, *, dt: D, first: T | None) -> None: ...

    def add_event(self, event: Event, *, n=None, dt=None, first=None) -> None:
        """Add an event to the simulation.

        Args:
            event: The event to add.
            n: The number of iterations between event triggers.
            dt: The time interval between event triggers.
            first: The first iteration or time to trigger the event.
        """
        if n is not None and dt is not None:
            raise ValueError("Cannot specify both n and dt.")
        elif n is not None and dt is None:
            self.every_n(n, event, first=first)
        elif n is None and dt is not None:
            self.every_dt(dt, event, first=first)
        else:
            raise ValueError("Either n or dt must be specified.")

    def add_output_writer(
        self,
        builder: AbstractOutputWriterBuilder,
        *,
        n: int | None = None,
        dt: D | None = None,
        first: int | T | None = None,
    ) -> None:
        """Add an output writer to the simulation.

        Args:
            writer: The output writer instance.
        """
        name = builder.name
        if name in self._output_writers:
            raise ValueError(f"Output writer '{name}' already exists.")

        kwargs = {
            "n": n,
            "dt": dt,
            "first": first,
        }
        self._output_writers[name] = (builder, kwargs)

    def build_simulation(self, nparticles: int) -> Simulation:
        """Build and return the Simulation.

        Args:
            nparticles: The number of particles in the simulation.
        """
        # build output writers, construct events and make mapping immutable
        output_writers = {}
        for name, (builder, kwargs) in self._output_writers.items():
            output_writers[name] = builder.build(nparticles)
            events = output_writers[name].create_events()
            for event in events:
                self.add_event(event, **kwargs)
        output_writers = types.MappingProxyType(output_writers)

        return Simulation(
            nparticles=nparticles,
            timestepper=self._timestepper,
            fieldset=self._fieldset,
            iteration_scheduler=self._iteration_scheduler,
            time_scheduler=self._time_scheduler,
            output_writers=output_writers,
        )
