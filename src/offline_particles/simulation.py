"""Submodule defining the top-level particle simulation class."""

import dataclasses
import itertools
import time
import types
import warnings
from collections.abc import Iterable, Mapping

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
from .kernels import BoundKernel, construct_validation_kernel_from_bbox
from .launcher import Launcher, Tinfo
from .output import AbstractOutputWriter, AbstractOutputWriterBuilder, Output
from .particles import Particles, ParticlesView
from .timestepping import Clock, D, T, Timestepper

DEFAULT_BBOX_HISTORY_SIZE = 256


@dataclasses.dataclass
class ParticleSet:
    """Class representing a set of particles."""

    name: str
    nparticles: int
    timestepper: Timestepper
    property_dtypes: dict[str, npt.DTypeLike] = dataclasses.field(default_factory=dict)
    include_validation_kernel: bool = True


class Simulation:
    """Class representing a particle simulation."""

    def __init__(
        self,
        clock: Clock,
        fieldset: Fieldset,
        timesteppers: Mapping[str, Timestepper],
        particles: Mapping[str, Particles],
        particles_view: Mapping[str, ParticlesView],
        recurring_iteration_scheduler: RecurringIterationScheduler,
        recurring_time_scheduler: RecurringTimeScheduler,
        at_iteration_scheduler: AtIterationScheduler,
        at_time_scheduler: AtTimeScheduler,
        output_writers: Mapping[str, AbstractOutputWriter],
        *,
        bbox_history_size: int = DEFAULT_BBOX_HISTORY_SIZE,
    ) -> None:
        """Construct the Simulation class.

        Parameters
        ----------
        clock : Clock
            The Clock governing simulation time and iteration.
        fieldset : Fieldset
            The Fieldset providing velocity and other field data.
        timesteppers : Mapping[str, Timestepper]
            Mapping of timestepper instances keyed by particle set name.
        particles : Mapping[str, Particles]
            Mapping of particle instances keyed by particle set name.
        particles_view : Mapping[str, ParticlesView]
            Mapping of particle view instances keyed by particle set name.
        recurring_iteration_scheduler : RecurringIterationScheduler
            Scheduler that fires events every N iterations.
        recurring_time_scheduler : RecurringTimeScheduler
            Scheduler that fires events every dt in time.
        at_iteration_scheduler : AtIterationScheduler
            Scheduler that fires events once at a specific iteration.
        at_time_scheduler : AtTimeScheduler
            Scheduler that fires events once at a specific time.
        output_writers : Mapping[str, AbstractOutputWriter]
            Mapping of output writer instances keyed by name.
        bbox_history_size : int, optional
            Number of bounding-box snapshots to retain for the launcher.

        Notes
        -----
            Use :class:`SimulationBuilder` to construct a simulation rather
            than instantiating this class directly.
        """
        self._clock = clock
        self._fieldset = fieldset
        self._timesteppers = timesteppers
        self._particles = particles
        self._particles_view = particles_view
        self._recurring_iteration_scheduler = recurring_iteration_scheduler
        self._recurring_time_scheduler = recurring_time_scheduler
        self._at_iteration_scheduler = at_iteration_scheduler
        self._at_time_scheduler = at_time_scheduler
        self._output_writers = output_writers

        # create launcher then register scalar data sources and set index padding
        self._launcher = Launcher(fieldset, history_size=bbox_history_size)
        self._launcher.register_scalar_data_sources_from_object(clock)
        for timestepper in self._timesteppers.values():
            self._launcher.register_scalar_data_sources_from_object(timestepper)
            self._launcher.set_index_padding(timestepper.index_padding)
        for event in self.events:
            self._launcher.register_scalar_data_sources_from_object(event)

        # stopping conditions
        # Default: stop at the chronological end of the time array
        self._iteration_stop = None
        self._time_stop = None
        self._wall_time_stop = None
        self.set_time_stop(self._clock.final_time)

        # store the current wall time
        self._wall_time_start = time.perf_counter_ns()

    # getters

    @property
    def fieldset(self) -> Fieldset:
        """Get the fieldset used in the simulation.

        Returns
        -------
        Fieldset
            The fieldset instance.
        """
        return self._fieldset

    @property
    def time(self) -> T:
        """Get the current simulation time.

        Returns
        -------
        T
            The current time of the simulation.
        """
        return self._clock.time

    @property
    def iteration(self) -> int:
        """Get the current simulation iteration.

        Returns
        -------
        int
            The current iteration of the simulation.
        """
        return self._clock.iteration

    @property
    def dt(self) -> D:
        """Get the timestep size.

        Returns
        -------
        D
            The size of each timestep in the simulation.
        """
        return self._clock.dt

    @property
    def tidx(self) -> np.float64:
        """Get the current timestep index.

        Returns
        -------
        np.float64
            The index of the current timestep.
        """
        return self._clock.tidx

    @property
    def time_unit(self) -> D:
        """Get the time unit used by the timestepper.

        Returns
        -------
        D
            The time unit.
        """
        return self._clock.time_unit

    @property
    def tinfo(self) -> Tinfo:
        """Get the current time information.

        Returns
        -------
        Tinfo
            The current time information named tuple.
        """
        return self._clock.tinfo

    @property
    def wall_time(self) -> np.timedelta64:
        """Get the elapsed wall time since the start of the simulation.

        Returns
        -------
        np.timedelta64
            The elapsed wall time.
        """
        nanoseconds = time.perf_counter_ns() - self._wall_time_start
        return np.timedelta64(nanoseconds, "ns")

    @property
    def index_padding(self) -> int:
        """Get the current index padding used by the launcher.

        Returns
        -------
        int
            The index padding.
        """
        return self._launcher.index_padding

    @property
    def forward_in_time(self) -> bool:
        """Check if the simulation is running forward in time.

        Returns
        -------
        bool
            True if the simulation is running forward in time, False otherwise.
        """
        return self._clock.forward_in_time

    @property
    def iteration_stop(self) -> int | None:
        """Get the iteration stopping condition.

        Returns
        -------
        int | None
            The iteration stopping condition, or None if not set.
        """
        return self._iteration_stop

    @property
    def time_stop(self) -> T | None:
        """Get the time stopping condition.

        Returns
        -------
        T | None
            The time stopping condition, or None if not set.
        """
        return self._time_stop

    @property
    def wall_time_stop(self) -> np.timedelta64 | None:
        """Get the wall time stopping condition.

        Returns
        -------
        np.timedelta64 | None
            The wall time stopping condition, or None if not set.
        """
        return self._wall_time_stop

    # setters

    def set_time(self, time: T) -> None:
        """Set the current simulation time.

        Parameters
        ----------
        time : T
            The new simulation time.
        """
        self._clock.set_time(time)

    def set_iteration(self, iteration: int) -> None:
        """Set the current simulation iteration.

        Parameters
        ----------
        iteration : int
            The new simulation iteration.
        """
        self._clock.set_iteration(iteration)

    def set_dt(self, dt: D) -> None:
        """Set the timestep size.

        Parameters
        ----------
        dt : D
            The new timestep size.
        """
        self._clock.set_dt(dt)

    def set_index_padding(self, index_padding: int, force: bool = False) -> None:
        """Set the index padding used by the launcher.

        Unless `force` is True, can only increase the index padding.
        """
        self._launcher.set_index_padding(index_padding, force=force)

    def set_iteration_stop(self, iteration: int | None) -> None:
        """Set the iteration stopping condition.

        Parameters
        ----------
        iteration : int | None
            The iteration to stop the simulation at, or None to disable.
        """
        self._iteration_stop = iteration

    def set_time_stop(self, time: T | None) -> None:
        """Set the time stopping condition.

        Parameters
        ----------
        time : T | None
            The time to stop the simulation at, or None to disable.

        Raises
        ------
        TypeError
            If the provided time is not compatible with the current simulation time.
        """
        if time is not None:
            # check time is compatible with current simulation time
            try:
                _ = time < self.time  # type: ignore
            except TypeError as e:
                raise TypeError(
                    f"Incompatible time type {type(time)} for simulation time type {type(self.time)}"
                ) from e
        self._time_stop = time

    def set_wall_time_stop(self, wall_time: np.timedelta64 | None) -> None:
        """Set the wall time stopping condition.

        Parameters
        ----------
        wall_time : np.timedelta64 | None
            The wall time to stop the simulation at, or None to disable.
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

        Parameters
        ----------
        iteration : int | None, optional
            The iteration to stop the simulation at, or None to disable.
        time : T | None, optional
            The time to stop the simulation at, or None to disable.
        wall_time : np.timedelta64 | None, optional
            The wall time to stop the simulation at, or None to disable.
        """
        self.set_iteration_stop(iteration)
        self.set_time_stop(time)
        self.set_wall_time_stop(wall_time)

    @property
    def particles(self) -> Mapping[str, ParticlesView]:
        """A view into the current particle data.

        Returns
        -------
        Mapping[str, ParticlesView]
            The current state of the particles in the simulation.
        """
        return types.MappingProxyType(self._particles_view)

    @property
    def recurring_iteration_scheduler(self) -> RecurringIterationScheduler:
        """Get the recurring iteration scheduler used in the simulation.

        Returns
        -------
        RecurringIterationScheduler
            The recurring iteration scheduler instance.
        """
        return self._recurring_iteration_scheduler

    @property
    def recurring_time_scheduler(self) -> RecurringTimeScheduler:
        """Get the recurring time scheduler used in the simulation.

        Returns
        -------
        RecurringTimeScheduler
            The recurring time scheduler instance.
        """
        return self._recurring_time_scheduler

    @property
    def at_iteration_scheduler(self) -> AtIterationScheduler:
        """Get the one-shot iteration scheduler used in the simulation.

        Returns
        -------
        AtIterationScheduler
            The one-shot iteration scheduler instance.
        """
        return self._at_iteration_scheduler

    @property
    def at_time_scheduler(self) -> AtTimeScheduler:
        """Get the one-shot time scheduler used in the simulation.

        Returns
        -------
        AtTimeScheduler
            The one-shot time scheduler instance.
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

        Returns
        -------
        SimulationState
            A class containing time information and views into the particle data.
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
        # run validation kernels
        for pset_name, particles in self._particles.items():
            self._timesteppers[pset_name].run_validation(particles, self._launcher, self._clock)
        # run all pre step kernels
        for pset_name, particles in self._particles.items():
            self._timesteppers[pset_name].run_pre_step(particles, self._launcher, self._clock)
        # run main step
        for pset_name, particles in self._particles.items():
            self._timesteppers[pset_name].run_step(particles, self._launcher, self._clock)
        # advance time
        self._clock.advance_time()
        # run all post step kernels
        for pset_name, particles in self._particles.items():
            self._timesteppers[pset_name].run_post_step(particles, self._launcher, self._clock)

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
            for pset_name, particles in self._particles.items():
                kernels = event.kernels.get(pset_name, ())
                for kernel in kernels:
                    self._launcher.launch_kernel(kernel, particles, self.tinfo)
            # invoke event function
            event(self.state)

    def run(self) -> None:
        """Run the particle simulation until a stopping condition is met.

        Raises
        ------
        ValueError
            If no valid stopping condition is set for the simulation.
        """
        # check we have at least one valid stopping condition
        valid_iteration_stop = self._iteration_stop is not None and self._iteration_stop > self.iteration
        valid_time_stop = self._time_stop is not None and (
            self._time_stop > self.time if self.forward_in_time else self._time_stop < self.time
        )
        valid_wall_time_stop = self._wall_time_stop is not None and self._wall_time_stop > self.wall_time
        if not (valid_iteration_stop or valid_time_stop or valid_wall_time_stop):
            raise ValueError("No valid stopping condition set for simulation.")

        # run initialisation kernels
        for pset_name, particles in self._particles.items():
            self._timesteppers[pset_name].run_initialisation(particles, self._launcher, self._clock)
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
        """Set the particles indices.

        Parameters
        ----------
        particle_set : str
            The name of the particle set to modify.
        zidx : npt.ArrayLike | None, optional
            The new z indices for the particles, or None to leave unchanged.
        yidx : npt.ArrayLike | None, optional
            The new y indices for the particles, or None to leave unchanged.
        xidx : npt.ArrayLike | None, optional
            The new x indices for the particles, or None to leave unchanged.

        Raises
        ------
        KeyError
            If the specified particle set does not exist in the simulation.
        """
        if particle_set not in self._particles:
            raise KeyError(f"Particle set '{particle_set}' not found in simulation.")
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

        Parameters
        ----------
        particle_set : str
            The name of the particle set to modify.
        property_name : str
            The name of the particle property to set.
        values : npt.ArrayLike
            The values to set the particle field to.

        Raises
        ------
        KeyError
            If the specified particle set does not exist in the simulation.
        """
        if particle_set not in self._particles:
            raise KeyError(f"Particle set '{particle_set}' not found in simulation.")

        particle_property = self._particles[particle_set][property_name]
        values_array = np.asarray(values, dtype=particle_property.dtype)
        values_array = np.broadcast_to(values_array, particle_property.shape)
        particle_property[:] = values_array

    def run_kernel(self, particle_set: str, kernel: BoundKernel) -> None:
        """Execute a kernel on the particles.

        Parameters
        ----------
        particle_set : str
            The name of the particle set to run the kernel on.
        kernel : BoundKernel
            The kernel to execute.

        Raises
        ------
        KeyError
            If the specified particle set does not exist in the simulation.
            Or if the kernel requires a particle property that is not available in the simulation.
        TypeError
            If a required particle property has an incompatible dtype. From :meth:`~ParticlePropertyDeclaration.validate_dtype`.
        """
        # get particles
        if particle_set not in self._particles:
            raise KeyError(f"Particle set '{particle_set}' not found in simulation.")
        particles = self._particles[particle_set]

        # check required particle properties are available
        for bound_name, declaration in kernel.particle_property_declarations.items():
            if bound_name not in particles.arrays:
                raise KeyError(
                    f"Particle property '{bound_name}' required by kernel is not available in the simulation."
                )
            declaration.validate_dtype(particles.arrays[bound_name].dtype)
        self._launcher.launch_kernel(kernel, particles, self.tinfo)


class SimulationBuilder:
    def __init__(
        self,
        clock: Clock,
        fieldset: Fieldset,
        *particle_sets: ParticleSet,
    ) -> None:
        """Class for building a Simulation.

        Parameters
        ----------
        clock : Clock
            The clock to use in the simulation.
        fieldset : Fieldset
            The fieldset to use in the simulation.
        *particle_sets : ParticleSet
            The particle sets to include in the simulation.

        Raises
        ------
        ValueError
            If particle set names are not unique.
        """
        self._clock = clock
        self._fieldset = fieldset

        # check particle set names are unique then store
        particle_set_names = [pset.name for pset in particle_sets]
        if len(particle_set_names) != len(set(particle_set_names)):
            raise ValueError("Particle set names must be unique.")
        self._particle_sets = list(particle_sets)

        # timesteppers
        self._timesteppers = {}
        for pset in particle_sets:
            timestepper = pset.timestepper
            # add validation kernel if requested - note this is cached so only constructed once per unique bbox
            if pset.include_validation_kernel:
                validation_kernel = construct_validation_kernel_from_bbox(self._fieldset.domain_bbox)
                timestepper.add_validation_kernels(validation_kernel)
            # store timestepper
            self._timesteppers[pset.name] = timestepper

        # events
        self._recurring_iteration_scheduler = RecurringIterationScheduler()
        self._recurring_time_scheduler = RecurringTimeScheduler(forward_in_time=self._clock.forward_in_time)
        self._at_iteration_scheduler = AtIterationScheduler()
        self._at_time_scheduler = AtTimeScheduler(forward_in_time=self._clock.forward_in_time)

        # output writers
        self._output_writer_builders: dict[str, AbstractOutputWriterBuilder] = {}
        self._output_writer_scheduling_args: dict[str, dict[str, ...]] = {}

    def collect_outputs(self) -> Mapping[str, list[Output]]:
        """Collect a list of outputs registered to particle sets across all writers.

        Returns
        -------
        Mapping[str, list[Output]]
            A mapping from particle set name to a list of outputs registered across all writers.
        """
        outputs_by_particle_set: dict[str, list[Output]] = {pset.name: [] for pset in self._particle_sets}
        for writer_builder in self._output_writer_builders.values():
            for (pset_name, _), output in writer_builder.outputs:
                outputs_by_particle_set[pset_name].append(output)
            for (pset_name, _), output in writer_builder.static_outputs:
                outputs_by_particle_set[pset_name].append(output)
        return types.MappingProxyType(outputs_by_particle_set)

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

        Parameters
        ----------
        n : int
            The interval in iterations between event triggers.
        event : Event
            The event to be added.
        first : int, optional
            The first iteration to trigger the event. Defaults to 0.

        Raises
        ------
        ValueError
            If n is not a positive integer.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if first is None:
            first = 0

        self._recurring_iteration_scheduler.register_event(first, n, event)

    def every_dt(self, dt: D, event: Event, *, first: T | None = None) -> None:
        """Add an event that triggers every dt time units.

        Parameters
        ----------
        dt : D
            The interval in time between event triggers.
        event : Event
            The event to be added.
        first : T, optional
            The first time to trigger the event (defaults to clock.time).

        Raises
        ------
        TypeError
            If dt or first are not compatible with the timestepper's time type.
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

        Parameters
        ----------
        time : T
            The time at which to trigger the event.
        event : Event
            The event to be triggered.

        Raises
        ------
        TypeError
            If the provided time is not compatible with the current simulation time.
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

        Parameters
        ----------
        event : Event
            The event to add.
        n : int | None, optional
            The number of iterations between event triggers.
        dt : D | None, optional
            The time interval between event triggers.
        first : int | T | None, optional
            When using ``n``, the first iteration (``int``) to trigger the event (defaults to 0).
            When using ``dt``, the first time (``T``) to trigger the event (defaults to the current clock time).

        Raises
        ------
        ValueError
            If neither or both of ``n`` and ``dt`` are specified.
        """
        match n, dt:
            case None, None:
                raise ValueError("Exactly one of n or dt must be specified.")
            case _, None:
                self.every_n(n, event, first=first)  # type: ignore[arg-type]
            case None, _:
                self.every_dt(dt, event, first=first)  # type: ignore[arg-type]
            case _:
                raise ValueError("Exactly one of n or dt must be specified.")

    def add_event(self, event: Event, *, at_iteration: int | None = None, at_time: T | None = None) -> None:
        """Add a one-shot event to the simulation.

        Exactly one of ``at_iteration`` or ``at_time`` must be specified.

        Parameters
        ----------
        event : Event
            The event to add.
        at_iteration : int | None, optional
            The specific iteration to trigger the event once.
        at_time : T | None, optional
            The specific time to trigger the event once.

        Raises
        ------
        ValueError
            If neither or both of ``at_iteration`` and ``at_time`` are specified.
        """
        match at_iteration, at_time:
            case None, None:
                raise ValueError("Exactly one of at_iteration or at_time must be specified.")
            case _, None:
                self.at_iteration(at_iteration, event)  # type: ignore[arg-type]
            case None, _:
                self.at_time(at_time, event)  # type: ignore[arg-type]
            case _:
                raise ValueError("Exactly one of at_iteration or at_time must be specified.")

    def add_output_writer(
        self,
        builder: AbstractOutputWriterBuilder,
        *,
        n: int | None = None,
        dt: D | None = None,
        first: int | T | None = None,
    ) -> None:
        """Add an output writer to the simulation.

        Parameters
        ----------
        builder : AbstractOutputWriterBuilder
            The output writer builder instance.
        n : int | None, optional
            The number of iterations between output writes (recurring).
        dt : D | None, optional
            The time interval between output writes (recurring).
        first : int | T | None, optional
            The first iteration or time to write output (used with n or dt).

        Raises
        ------
        ValueError
            If an output writer with the same name already exists.
            If neither or both of ``n`` and ``dt`` are specified.
        """
        name = builder.name
        if name in self._output_writer_builders:
            raise ValueError(f"Output writer '{name}' already exists.")

        match n, dt:
            case None, None:
                raise ValueError("Exactly one of n or dt must be specified.")
            case _, None:
                if first is None:
                    first = 0
                scheduling_args = {"n": n, "first": first}  # type: ignore
            case None, _:
                if first is None:
                    first = self._clock.time
                scheduling_args = {"dt": dt, "first": first}  # type: ignore
            case _:
                raise ValueError("Exactly one of n or dt must be specified.")

        self._output_writer_builders[name] = builder
        self._output_writer_scheduling_args[name] = scheduling_args

    def build_simulation(self, *, bbox_history_size: int = DEFAULT_BBOX_HISTORY_SIZE) -> Simulation:
        """Build and return the Simulation.

        Parameters
        ----------
        bbox_history_size : int, optional
            The number of bounding-box snapshots to retain for the launcher (default: ``DEFAULT_BBOX_HISTORY_SIZE``).

        Returns
        -------
        Simulation
            The constructed Simulation instance.
        """
        # to build the particles we need to gather all the kernels
        # first collect the outputs
        outputs = self.collect_outputs()

        # then initialise the particles dict
        particles = {}
        particles_view = {}

        # keep track of total kernel count across all particle sets for warning about bbox history size
        kernel_count = 0

        # loop over particle sets
        for pset in self._particle_sets:
            name = pset.name
            nparticles = pset.nparticles

            # gather kernels from timesteppers
            kernels = list(self._timesteppers[name].kernels)
            # then gather kernels from events
            for event in self.events:
                kernels.extend(event.kernels.get(name, ()))
            # finally from outputs
            for output in outputs.get(name, ()):
                kernels.extend(output.kernels)

            kernel_count += len(kernels)

            particles[name] = Particles.build_from_kernels(nparticles, pset.property_dtypes, kernels)
            particles_view[name] = ParticlesView(particles[name])

        # print a warning if kernel count is larger that history size
        if kernel_count > bbox_history_size:
            warnings.warn(
                f"Number of kernels ({kernel_count}) "
                f"exceeds bbox history size ({bbox_history_size}). "
                "Consider increasing the history size for better performance.",
                RuntimeWarning,
            )

        # next build the output writers and register their events
        built_output_writers = {}
        time_type = self._clock.time_array.dtype

        for name, builder in self._output_writer_builders.items():
            built_output_writers[name] = builder.build(particles_view, time_type)

            # register recurring output events
            scheduling_args = self._output_writer_scheduling_args.get(name, {})
            for event in built_output_writers[name].create_output_events():
                self.add_recurring_event(event, **scheduling_args)

            # register static output events once at iteration 0
            for event in built_output_writers[name].create_static_output_events():
                self.at_iteration(0, event)

        output_writers = types.MappingProxyType(built_output_writers)

        return Simulation(
            clock=self._clock,
            fieldset=self._fieldset,
            timesteppers=types.MappingProxyType(self._timesteppers),
            particles=types.MappingProxyType(particles),
            particles_view=types.MappingProxyType(particles_view),
            recurring_iteration_scheduler=self._recurring_iteration_scheduler,
            recurring_time_scheduler=self._recurring_time_scheduler,
            at_iteration_scheduler=self._at_iteration_scheduler,
            at_time_scheduler=self._at_time_scheduler,
            output_writers=output_writers,
            bbox_history_size=bbox_history_size,
        )
