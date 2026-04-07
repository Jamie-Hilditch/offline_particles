"""Submodule defining the top-level particle simulation class."""

import dataclasses
import itertools
import time
import types
import warnings
from typing import Iterable, Mapping

import numpy as np
import numpy.typing as npt

from .events import (
    AtIterationScheduler,
    AtTimeScheduler,
    Event,
    RecurringIterationScheduler,
    RecurringTimeScheduler,
    SimulationState,
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
        recurring_iteration_scheduler: RecurringIterationScheduler,
        recurring_time_scheduler: RecurringTimeScheduler,
        at_iteration_scheduler: AtIterationScheduler,
        at_time_scheduler: AtTimeScheduler,
        output_writers: Mapping[str, AbstractOutputWriter],
        *,
        bbox_history_size: int = DEFAULT_BBOX_HISTORY_SIZE,
    ) -> None:
        """Initialize the Simulation.

        Args:
            clock: The Clock governing simulation time and iteration.
            fieldset: The Fieldset providing velocity and other field data.
            particle_sets: List of ParticleSet instances defining particle groups.
            recurring_iteration_scheduler: Scheduler that fires events every N iterations.
            recurring_time_scheduler: Scheduler that fires events every dt in time.
            at_iteration_scheduler: Scheduler that fires events once at a specific iteration.
            at_time_scheduler: Scheduler that fires events once at a specific time.
            output_writers: Mapping of output writer instances keyed by name.
            bbox_history_size: Number of bounding-box snapshots to retain for the launcher.
        """
        self._clock = clock
        self._fieldset = fieldset
        self._recurring_iteration_scheduler = recurring_iteration_scheduler
        self._recurring_time_scheduler = recurring_time_scheduler
        self._at_iteration_scheduler = at_iteration_scheduler
        self._at_time_scheduler = at_time_scheduler
        self._output_writers = output_writers

        # create launcher and register kernel data functions
        self._launcher = Launcher(fieldset, history_size=bbox_history_size)
        self._launcher.register_scalar_data_sources_from_object(clock)
        for event in self.events:
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
            for event in self.events:
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
    def forward_in_time(self) -> bool:
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
        if time is not None:
            # check time is compatible with current simulation time
            try:
                time < self.time  # type: ignore
            except TypeError as e:
                raise TypeError(
                    f"Incompatible time type {type(time)} for simulation time type {type(self.time)}"
                ) from e
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
    def recurring_iteration_scheduler(self) -> RecurringIterationScheduler:
        """Get the recurring iteration scheduler used in the simulation.

        Returns:
            RecurringIterationScheduler: The recurring iteration scheduler instance.
        """
        return self._recurring_iteration_scheduler

    @property
    def recurring_time_scheduler(self) -> RecurringTimeScheduler:
        """Get the recurring time scheduler used in the simulation.

        Returns:
            RecurringTimeScheduler: The recurring time scheduler instance.
        """
        return self._recurring_time_scheduler

    @property
    def at_iteration_scheduler(self) -> AtIterationScheduler:
        """Get the one-shot iteration scheduler used in the simulation.

        Returns:
            AtIterationScheduler: The one-shot iteration scheduler instance.
        """
        return self._at_iteration_scheduler

    @property
    def at_time_scheduler(self) -> AtTimeScheduler:
        """Get the one-shot time scheduler used in the simulation.

        Returns:
            AtTimeScheduler: The one-shot time scheduler instance.
        """
        return self._at_time_scheduler

    @property
    def events(self) -> Iterable[Event]:
        """All events registered across all schedulers."""
        return itertools.chain(
            self._recurring_iteration_scheduler.events,
            self._at_iteration_scheduler.events,
            self._recurring_time_scheduler.events,
            self._at_time_scheduler.events,
        )

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
        events_to_be_invoked = itertools.chain(
            self._recurring_iteration_scheduler(self.iteration),
            self._at_iteration_scheduler(self.iteration),
            self._recurring_time_scheduler(self.time),
            self._at_time_scheduler(self.time),
        )
        for event in events_to_be_invoked:
            # launch kernels
            for name, particles in self._particles.items():
                kernels = event.kernels.get(name, ())
                for kernel in kernels:
                    self._launcher.launch_kernel(kernel, particles, self.tinfo)
            # invoke event function
            event(self.state)

    def run(self) -> None:
        """Run the particle simulation until a stopping condition is met."""
        # The end of the time array is always a valid default stopping condition
        time_array = self._clock.time_array
        valid_time_array_stop = (
            time_array[-1] > self.time if self.forward_in_time else time_array[0] < self.time
        )
        # check we have at least one valid stopping condition
        valid_iteration_stop = self._iteration_stop is not None and self._iteration_stop > self.iteration
        valid_time_stop = self._time_stop is not None and (
            self._time_stop > self.time if self.forward_in_time else self._time_stop < self.time
        )
        valid_wall_time_stop = self._wall_time_stop is not None and self._wall_time_stop > self.wall_time
        if not (valid_iteration_stop or valid_time_stop or valid_wall_time_stop or valid_time_array_stop):
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
            # Default stopping condition: end of the time array.
            # Stop before attempting a step that would take the simulation out of bounds.
            if self.forward_in_time and self.time + self.dt > time_array[-1]:  # type: ignore[operator]
                break
            if not self.forward_in_time and self.time + self.dt < time_array[0]:  # type: ignore[operator]
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
        self._recurring_iteration_scheduler = RecurringIterationScheduler()
        self._recurring_time_scheduler = RecurringTimeScheduler(forward_in_time=self._clock.forward_in_time)
        self._at_iteration_scheduler = AtIterationScheduler()
        self._at_time_scheduler = AtTimeScheduler(forward_in_time=self._clock.forward_in_time)

        # output writers
        self._output_writers: dict[str, tuple[AbstractOutputWriterBuilder, dict[str, ...]]] = dict()

    @property
    def events(self) -> Iterable[Event]:
        """All events registered across all schedulers."""
        return itertools.chain(
            self._recurring_iteration_scheduler.events,
            self._at_iteration_scheduler.events,
            self._recurring_time_scheduler.events,
            self._at_time_scheduler.events,
        )

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
        self._recurring_iteration_scheduler.register_event(first, n, event)

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

        self._recurring_time_scheduler.register_event(first, dt, event)

    def at_iteration(self, iteration: int, event: Event) -> None:
        """Add an event that triggers once at the given iteration.

        Args:
            iteration (int): The iteration at which to trigger the event.
            event (Event): The event to be triggered.
        """
        self._at_iteration_scheduler.register_event(iteration, event)

    def at_time(self, time: T, event: Event) -> None:
        """Add an event that triggers once at the given time.

        Args:
            time (T): The time at which to trigger the event.
            event (Event): The event to be triggered.
        """
        # check time is compatible with clock time
        clock_time = self._clock.time
        try:
            _ = time < clock_time  # type: ignore
        except TypeError as e:
            raise TypeError(f"Incompatible time type {type(time)} for timestepper time type {type(clock_time)}") from e

        self._at_time_scheduler.register_event(time, event)

    def add_recurring_event(
        self, event: Event, *, n: int | None = None, dt: D | None = None, first: int | T | None = None
    ) -> None:
        """Add a recurring event to the simulation.

        Exactly one of ``n`` or ``dt`` must be specified.

        Args:
            event: The event to add.
            n: The number of iterations between event triggers.
            dt: The time interval between event triggers.
            first: When using ``n``, the first iteration (``int``) to trigger the event (defaults to 0).
                When using ``dt``, the first time (``T``) to trigger the event (defaults to the current clock time).
        """
        if (n is None) == (dt is None):
            raise ValueError("Exactly one of n or dt must be specified.")
        if n is not None:
            self.every_n(n, event, first=first)  # type: ignore[arg-type]
        else:
            self.every_dt(dt, event, first=first)  # type: ignore[arg-type]

    def add_event(self, event: Event, *, at_iteration: int | None = None, at_time: T | None = None) -> None:
        """Add a one-shot event to the simulation.

        Exactly one of ``at_iteration`` or ``at_time`` must be specified.

        Args:
            event: The event to add.
            at_iteration: The specific iteration to trigger the event once.
            at_time: The specific time to trigger the event once.
        """
        if (at_iteration is None) == (at_time is None):
            raise ValueError("Exactly one of at_iteration or at_time must be specified.")
        if at_iteration is not None:
            self.at_iteration(at_iteration, event)
        else:
            self.at_time(at_time, event)  # type: ignore[arg-type]

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
            builder: The output writer builder instance.
            n: The number of iterations between output writes (recurring).
            dt: The time interval between output writes (recurring).
            first: The first iteration or time to write output (used with n or dt).
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
            for event in output_writers[name].create_output_events():
                self.add_recurring_event(event, n=kwargs.get("n"), dt=kwargs.get("dt"), first=kwargs.get("first"))
            # register static output events once at iteration 0
            for event in output_writers[name].create_static_output_events():
                self.at_iteration(0, event)
        output_writers = types.MappingProxyType(output_writers)

        return Simulation(
            clock=self._clock,
            fieldset=self._fieldset,
            particle_sets=self._particle_sets,
            recurring_iteration_scheduler=self._recurring_iteration_scheduler,
            recurring_time_scheduler=self._recurring_time_scheduler,
            at_iteration_scheduler=self._at_iteration_scheduler,
            at_time_scheduler=self._at_time_scheduler,
            output_writers=output_writers,
        )
