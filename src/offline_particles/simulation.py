"""Submodule defining the top-level particle simulation class."""

import dataclasses
import itertools
import time
import types
import warnings
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
from .kernels import BoundKernel, get_required_particle_property_dtypes
from .launcher import Launcher, Tinfo
from .output import AbstractOutputWriter, AbstractOutputWriterBuilder
from .particles import Particles, ParticlesView
from .timestepping import Clock, Timestepper

type T = np.float64 | np.datetime64
type D = np.float64 | np.timedelta64

DEFAULT_BBOX_HISTORY_SIZE = 256


@dataclasses.dataclass
class ParticleSet:
    """Class representing a set of particles."""

    name: str
    nparticles: int
    timestepper: Timestepper


class Simulation:
    """Class representing a particle simulation."""

    def __init__(
        self,
        clock: Clock,
        fieldset: Fieldset,
        particle_sets: list[ParticleSet],
        iteration_scheduler: IterationScheduler,
        time_scheduler: TimeScheduler,
        output_writers: Mapping[str, AbstractOutputWriter],
        *,
        bbox_history_size: int = DEFAULT_BBOX_HISTORY_SIZE,
    ) -> None:
        """Initialize the Simulation.

        Args:
            builder: The SimulationBuilder used to configure the simulation.
        """
        self._clock = clock
        self._fieldset = fieldset
        self._iteration_scheduler = iteration_scheduler
        self._time_scheduler = time_scheduler
        self._output_writers = output_writers

        # create launcher and register kernel data functions
        self._launcher = Launcher(fieldset, history_size=bbox_history_size)
        self._launcher.register_scalar_data_sources_from_object(clock)
        for event in self._iteration_scheduler.events:
            self._launcher.register_scalar_data_sources_from_object(event)
        for event in self._time_scheduler.events:
            self._launcher.register_scalar_data_sources_from_object(event)

        # check particle set names are unique
        particle_set_names = [pset.name for pset in particle_sets]
        if len(particle_set_names) != len(set(particle_set_names)):
            raise ValueError("Particle set names must be unique.")

        # store timesteppers by name
        self._timesteppers = {pset.name: pset.timestepper for pset in particle_sets}
        for timestepper in self._timesteppers.values():
            self._launcher.set_index_padding(timestepper.index_padding)

        # now build particles
        self._particles = {}
        self._particles_view = {}

        # build particles
        kernel_count = 0
        for pset in particle_sets:
            name = pset.name
            nparticles = pset.nparticles
            # gather kernels
            kernels = list(pset.timestepper.kernels)
            for event in self._iteration_scheduler.events:
                kernels.extend(event.kernels.get(name, ()))
            for event in self._time_scheduler.events:
                kernels.extend(event.kernels.get(name, ()))
            kernel_count += len(kernels)

            # then merge required particle properties from all kernels
            particle_property_dtypes = get_required_particle_property_dtypes(*kernels)
            self._particles[name] = Particles(nparticles, **particle_property_dtypes)
            self._particles_view[name] = ParticlesView(self._particles[name])

        # print a warning if kernel count is larger that history size
        if kernel_count > bbox_history_size:
            warnings.warn(
                f"Number of kernels ({kernel_count}) "
                f"exceeds bbox history size ({bbox_history_size}). "
                "Consider increasing the history size for better performance.",
                RuntimeWarning,
            )

        # store the current wall time
        self._wall_time_start = time.perf_counter_ns()

        # stopping conditions
        self._iteration_stop = None
        self._time_stop = None
        self._wall_time_stop = None

    # getters

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
        return self._clock.time

    @property
    def iteration(self) -> int:
        """Get the current simulation iteration.

        Returns:
            int: The current iteration of the simulation.
        """
        return self._clock.iteration

    @property
    def dt(self) -> np.float64 | np.timedelta64:
        """Get the timestep size.

        Returns:
            float: The size of each timestep in the simulation.
        """
        return self._clock.dt

    @property
    def tidx(self) -> np.float64:
        """Get the current timestep index.

        Returns:
            float: The index of the current timestep.
        """
        return self._clock.tidx

    @property
    def time_unit(self) -> np.float64 | np.timedelta64:
        """Get the time unit used by the timestepper.

        Returns:
            The time unit.
        """
        return self._clock.time_unit

    @property
    def tinfo(self) -> Tinfo:
        """Get the current time information.

        Returns:
            Tinfo: The current time information named tuple.
        """
        return self._clock.tinfo

    @property
    def wall_time(self) -> np.timedelta64:
        """Get the elapsed wall time since the start of the simulation.

        Returns:
            float: The elapsed wall time in seconds.
        """
        nanoseconds = time.perf_counter_ns() - self._wall_time_start
        return np.timedelta64(nanoseconds, "ns")

    @property
    def index_padding(self) -> int:
        """Get the current index padding used by the launcher.

        Returns:
            int: The index padding.
        """
        return self._launcher.index_padding

    @property
    def forward_in_time(self) -> np.bool:
        """Check if the simulation is running forward in time.

        Returns:
            bool: True if the simulation is running forward in time, False otherwise.
        """
        return self._clock.forward_in_time

    @property
    def iteration_stop(self) -> int | None:
        """Get the iteration stopping condition.

        Returns:
            int | None: The iteration stopping condition, or None if not set.
        """
        return self._iteration_stop

    @property
    def time_stop(self) -> T | None:
        """Get the time stopping condition.

        Returns:
            T | None: The time stopping condition, or None if not set.
        """
        return self._time_stop

    @property
    def wall_time_stop(self) -> np.timedelta64 | None:
        """Get the wall time stopping condition.

        Returns:
            np.timedelta64 | None: The wall time stopping condition, or None if not set.
        """
        return self._wall_time_stop

    # setters

    def set_time(self, time: T) -> None:
        """Set the current simulation time.

        Args:
            time: The new simulation time.
        """
        self._clock.set_time(time)

    def set_iteration(self, iteration: int) -> None:
        """Set the current simulation iteration.

        Args:
            iteration: The new simulation iteration.
        """
        self._clock.set_iteration(iteration)

    def set_dt(self, dt: D) -> None:
        """Set the timestep size.

        Args:
            dt: The new timestep size.
        """
        self._clock.set_dt(dt)

    def set_index_padding(self, index_padding: int, force: bool = False) -> None:
        """Set the index padding used by the launcher.

        Unless `force` is True, can only increase the index padding.
        """
        self._launcher.set_index_padding(index_padding, force=force)

    def set_iteration_stop(self, iteration: int | None) -> None:
        """Set the iteration stopping condition.

        Args:
            iteration: The iteration to stop the simulation at, or None to disable.
        """
        self._iteration_stop = iteration

    def set_time_stop(self, time: T | None) -> None:
        """Set the time stopping condition.

        Args:
            time: The time to stop the simulation at, or None to disable.
        """
        # check time is compatible with current simulation time
        try:
            time < self.time  # type: ignore
        except TypeError as e:
            raise TypeError(f"Incompatible time type {type(time)} for simulation time type {type(self.time)}") from e
        self._time_stop = time

    def set_wall_time_stop(self, wall_time: np.timedelta64 | None) -> None:
        """Set the wall time stopping condition.

        Args:
            wall_time: The wall time to stop the simulation at, or None to disable.
        """
        self._wall_time_stop = wall_time

    def set_stopping_conditions(
        self,
        *,
        iteration: int | None = None,
        time: T | None = None,
        wall_time: np.timedelta64 | None = None,
    ) -> None:
        """Set the stopping conditions for the simulation.

        Args:
            iteration: The iteration to stop the simulation at, or None to disable.
            time: The time to stop the simulation at, or None to disable.
            wall_time: The wall time to stop the simulation at, or None to disable.
        """
        self.set_iteration_stop(iteration)
        self.set_time_stop(time)
        self.set_wall_time_stop(wall_time)

    @property
    def particles(self) -> Mapping[str, ParticlesView]:
        """A view into the current particle data.

        Returns:
            The current state of the particles in the simulation.
        """
        return types.MappingProxyType(self._particles_view)

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
            SimulationState: A named tuple containing time, dt, tidx, wall_time and particles.
        """
        return SimulationState(
            time=self.time,
            dt=self.dt,
            tidx=self.tidx,
            iteration=self.iteration,
            wall_time=self.wall_time,
            particles=self._particles_view,
        )

    # running the simulation

    def step(self) -> None:
        """Advance the particle simulation by one timestep."""
        # run all pre step kernels
        for name, particles in self._particles.items():
            self._timesteppers[name].run_pre_step(particles, self._launcher, self._clock)
        # run main step
        for name, particles in self._particles.items():
            self._timesteppers[name].run_step(particles, self._launcher, self._clock)
        # advance time
        self._clock.advance_time()
        # run all post step kernels
        for name, particles in self._particles.items():
            self._timesteppers[name].run_post_step(particles, self._launcher, self._clock)

    def _invoke_events(self) -> None:
        """Invoke any scheduled events at the current time or iteration."""
        for event in itertools.chain(self._iteration_scheduler(self.iteration), self._time_scheduler(self.time)):
            # launch kernels
            for name, particles in self._particles.items():
                kernels = event.kernels.get(name, ())
                for kernel in kernels:
                    self._launcher.launch_kernel(kernel, particles, self.tinfo)
            # invoke event function
            event(self.state)

    def run(self) -> None:
        """Run the particle simulation until a stopping condition is met."""
        # check we have at least one valid stopping condition
        valid_iteration_stop = self._iteration_stop is not None and self._iteration_stop > self.iteration
        valid_time_stop = self._time_stop is not None and (
            self._time_stop > self.time if self.forward_in_time else self._time_stop < self.time
        )
        valid_wall_time_stop = self._wall_time_stop is not None and self._wall_time_stop > self.wall_time
        if not (valid_iteration_stop or valid_time_stop or valid_wall_time_stop):
            raise ValueError("No valid stopping condition set for simulation.")

        # run initialisation kernels
        for name, particles in self._particles.items():
            self._timesteppers[name].run_initialisation(particles, self._launcher, self._clock)
        # invoke events at initial time / iteration
        self._invoke_events()

        while True:
            # check stopping conditions
            if self._iteration_stop is not None and self.iteration >= self._iteration_stop:
                break
            if self._time_stop is not None and (
                self.time >= self._time_stop if self.forward_in_time else self.time <= self._time_stop
            ):
                break
            if self._wall_time_stop is not None and self.wall_time >= self._wall_time_stop:
                break

            # advance one timestep
            self.step()

            # invoke events
            self._invoke_events()

    # initialization

    def set_indices(
        self,
        particle_set: str,
        *,
        zidx: npt.ArrayLike | None = None,
        yidx: npt.ArrayLike | None = None,
        xidx: npt.ArrayLike | None = None,
    ) -> None:
        """Set the particles indices."""

        if particle_set not in self._particles:
            raise ValueError(f"Particle set '{particle_set}' not found in simulation.")
        particles = self._particles[particle_set]

        # first make the inputs compatible with the particle arrays
        # allow this to error if the shapes / types are incompatible
        # before modifying any particle data
        if zidx is not None:
            zidx = np.asarray(zidx, dtype=np.float64)
            zidx = np.broadcast_to(zidx, particles.zidx.shape)

        if yidx is not None:
            yidx = np.asarray(yidx, dtype=np.float64)
            yidx = np.broadcast_to(yidx, particles.yidx.shape)

        if xidx is not None:
            xidx = np.asarray(xidx, dtype=np.float64)
            xidx = np.broadcast_to(xidx, particles.xidx.shape)

        # now set the indices
        if zidx is not None:
            particles.zidx[:] = zidx
        if yidx is not None:
            particles.yidx[:] = yidx
        if xidx is not None:
            particles.xidx[:] = xidx

    def set_particle_property(
        self,
        particle_set: str,
        property_name: str,
        values: npt.ArrayLike,
    ) -> None:
        """Set a particle property to the given values.

        Args:
            particle_set: The name of the particle set to modify.
            property_name: The name of the particle property to set.
            values: The values to set the particle field to.
        """
        if particle_set not in self._particles:
            raise ValueError(f"Particle set '{particle_set}' not found in simulation.")

        particle_property = self._particles[particle_set][property_name]
        values_array = np.asarray(values, dtype=particle_property.dtype)
        values_array = np.broadcast_to(values_array, particle_property.shape)
        particle_property[:] = values_array

    def run_kernel(self, name: str, kernel: BoundKernel) -> None:
        """Execute a kernel on the particles.

        Args:
            name: The name of the particle set to run the kernel on.
            kernel: The kernel to execute.
        """
        # get particles
        if name not in self._particles:
            raise ValueError(f"Particle set '{name}' not found in simulation.")
        particles = self._particles[name]

        # check required particle properties are available
        required_property_dtypes = get_required_particle_property_dtypes(kernel)
        for binding, dtype in required_property_dtypes.items():
            if binding not in particles.arrays:
                raise ValueError(
                    f"Particle property '{binding}' required by kernel is not available in the simulation."
                )
            if particles.arrays[binding].dtype != dtype:
                raise TypeError(
                    f"Particle property '{binding}' has dtype {particles.arrays[binding].dtype}, "
                    f"but kernel declares dtype {dtype}."
                )
        self._launcher.launch_kernel(kernel, particles, self.tinfo)


class SimulationBuilder:
    def __init__(
        self,
        clock: Clock,
        fieldset: Fieldset,
        *particle_sets: ParticleSet,
    ) -> None:
        """Class for building a Simulation.

        Args:
            clock: The clock to use in the simulation.
            fieldset: The fieldset to use in the simulation.
            particle_sets: The particle sets to include in the simulation.
        """
        self._clock = clock
        self._fieldset = fieldset
        self._particle_sets = list(particle_sets)

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
            first (T): The first time to trigger the event (defaults to clock.time).
        """
        # set default first time
        clock_time = self._clock.time
        if first is None:
            first = clock_time

        # check times are compatible
        try:
            _ = clock_time + dt  # type: ignore
        except TypeError as e:
            raise TypeError(f"Incompatible dt type {type(dt)} for timestepper time type {type(clock_time)}") from e
        try:
            _ = first + dt  # type: ignore
        except TypeError as e:
            raise TypeError(
                f"Incompatible first type {type(first)} for timestepper time type {type(clock_time)}"
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

    def build_simulation(self) -> Simulation:
        """Build and return the Simulation."""
        # build output writers, construct events and make mapping immutable
        output_writers = {}
        time_type = self._clock.time_array.dtype
        nparticles = {pset.name: pset.nparticles for pset in self._particle_sets}
        for name, (builder, kwargs) in self._output_writers.items():
            output_writers[name] = builder.build(nparticles, time_type)
            events = output_writers[name].create_events()
            for event in events:
                self.add_event(event, **kwargs)
        output_writers = types.MappingProxyType(output_writers)

        return Simulation(
            clock=self._clock,
            fieldset=self._fieldset,
            particle_sets=self._particle_sets,
            iteration_scheduler=self._iteration_scheduler,
            time_scheduler=self._time_scheduler,
            output_writers=output_writers,
        )
