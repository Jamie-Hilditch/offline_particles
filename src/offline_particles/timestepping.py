"""Submodule for timestepping classes.

Types
~~~~~

.. list-table::
   :header-rows: 0
   :widths: 10 45 45

   * - :py:data:`T`
     - ``np.floating | np.datetime64``
     - Supported time types.
   * - :py:data:`D`
     - ``np.floating | np.timedelta64``
     - Supported time increment types.
   * - :py:data:`DLike`
     - ``np.floating | np.timedelta64 | float | int``
     - Accepted time increment input types. Converted to :py:data:`D` internally."""

import abc
import itertools
from typing import Iterator

import numpy as np
import numpy.typing as npt

from .kernels import BoundKernel
from .kernels.timestepping import (
    construct_ab2_update_kernel,
    construct_ab3_update_kernel,
    construct_ab_bump_status_kernel,
    construct_ab_initialisation_kernel,
)
from .launcher import Launcher, ScalarSource, Time_info, Tinfo
from .particles import Particles

__all__ = [
    "ABTimestepper",
    "Clock",
    "D",
    "DLike",
    "RK2Timestepper",
    "T",
    "Timestepper",
]

# Supported time and time increment types
# We need to ensure these are kept up to date in the module docstring.
type T = np.floating | np.datetime64
type D = np.floating | np.timedelta64
type DLike = np.floating | np.timedelta64 | float | int


def _regularise_DLike(dt: DLike) -> D:
    """Regularise a DLike time increment into a D time increment."""
    match dt:
        case np.floating() | np.timedelta64():
            return dt
        case float() | int():
            return np.float64(dt)
        case _:
            raise TypeError(f"dt must be a np.floating, np.timedelta64, or float, got {type(dt)!r}")


def _convert_DLike(dt: DLike, time_unit: D) -> D:
    """Convert a DLike time increment into a D time increment of the same type as time_unit."""
    _dt = _regularise_DLike(dt)
    match _dt, time_unit:
        # dimensionless time: allow any floating type, convert to time_unit type
        case np.floating(), np.floating():
            return _dt.astype(time_unit.dtype)
        # dimensional time: require dt to be same type as time_unit
        case np.timedelta64(), np.timedelta64():
            return _dt.astype(time_unit.dtype)
        case _:
            raise TypeError(
                f"dt={dt!r} regularised to {_dt!r} which is not compatible with time_unit={time_unit!r}. "
                f"Non-dimensional time requires np.floating, "
                f"dimensional time requires np.timedelta64."
            )


class Clock:
    """Class keeping time for a simulation.

    Args:
        time_array: strictly increasing 1D array of time values.
        dt: The time step. Must be positive for forward integration,
            negative for backward integration.
        time_unit: The time unit. Determines the type of time increments. Required for dimensional time,
            defaults to np.float64(1.0) for dimensionless time.

    Raises:
        ValueError: If ``time_array`` is not 1D, has fewer than 2 elements,
            or is not strictly increasing.
        ValueError: If ``time_unit`` is not positive.
        ValueError: If ``dt`` has the wrong sign for the clock direction.
        TypeError: If ``dt`` or ``time_unit`` has an unsupported type, or if
            ``dt`` is incompatible with ``time_unit`` (for example, dimensional
            time requires ``np.timedelta64`` increments, while non-dimensional
            time requires floating-point increments).

    Note:
        The clock direction (forward or backward in time) is determined by
        the sign of ``dt`` at construction time and cannot be changed afterwards.

    Examples:
        >>> time_array = np.array([0, 1, 2, 3], dtype=np.float64)
        >>> dt = np.float64(0.5)
        >>> Clock(time_array, dt)
        Clock(dt=np.float64(0.5), time_unit=np.float64(1.0))
    """

    # scalar data sources
    _dt_scalar = ScalarSource("_dt", lambda self, tinfo: self._normalised_dt)
    _time_scalar = ScalarSource("_time", lambda self, tinfo: self.time)
    _tidx_scalar = ScalarSource("_tidx", lambda self, tinfo: self._tidx)
    _iteration_scalar = ScalarSource("_iteration", lambda self, tinfo: self._iteration)

    def __init__(
        self,
        time_array: npt.NDArray[T],
        dt: DLike,
        *,
        time_unit: DLike = np.float64(1.0),
    ) -> None:
        # validate time_array shape and length
        if time_array.ndim != 1:
            raise ValueError("time_array must be 1D.")
        if len(time_array) < 2:
            raise ValueError("time_array must have at least 2 elements.")
        # check time_array is strictly increasing
        if np.any(time_array[1:] <= time_array[:-1]):  # type: ignore[operator]
            raise ValueError("time_array must be strictly increasing.")
        self._time_array = time_array
        # precompute maximum searchsorted index: clamps idx so idx+1 is always valid
        self._max_tidx = len(time_array) - 2

        # determine increment dtype
        time_unit = _regularise_DLike(time_unit)
        if not (time_unit > time_unit * 0):
            raise ValueError("time_unit must be positive.")
        self._time_unit: D = time_unit

        # determine clock direction from sign of dt
        dt = _regularise_DLike(dt)
        if dt == dt * 0:
            raise ValueError("dt cannot be zero.")
        self._forward_in_time: bool = dt > 0 * dt

        # set dt (validates sign and computes normalised dt)
        self.set_dt(dt)

        # initialise time, time_index and iteration
        # use first_time so backward clocks start at time_array[-1]
        self.set_time(self.first_time)
        self.set_iteration(0)

    def __repr__(self) -> str:
        return f"Clock(dt={self.dt!r}, time_unit={self.time_unit!r})"

    def get_time_index(self, time: T) -> np.float64:
        """Get the time index corresponding to the given time.

        Args:
            time: The time to get the index for.

        Returns:
            float64: The time index corresponding to the given time.

        Raises:
            ValueError: If time is out of bounds of the time array.
            TypeError (from numpy): If time is not compatible with the time array.
        """
        time_array = self._time_array
        if time < time_array[0] or time > time_array[-1]:
            raise ValueError("Time is out of bounds of the time array.")

        idx = np.searchsorted(time_array, time, side="right") - 1
        # Clamp idx so that idx+1 is always a valid index (handles time == time_array[-1])
        if idx > self._max_tidx:
            idx = self._max_tidx
        t0 = time_array[idx]
        t1 = time_array[idx + 1]
        fraction = (time - t0) / (t1 - t0)
        return idx + fraction

    def set_dt(self, dt: DLike) -> None:
        """Set the time step."""
        dt = _convert_DLike(
            dt, self.time_unit
        )  # validates dt is compatible with time_unit and converts to correct type

        # validate sign of dt using zero of the same type as dt
        if self._forward_in_time and not (dt > dt * 0):
            raise ValueError("dt must be positive for forward-in-time integration.")
        if not self._forward_in_time and not (dt < dt * 0):
            raise ValueError("dt must be negative for backward-in-time integration.")

        # set normalised dt for internal use
        self._normalised_dt = np.float64(dt / self.time_unit)  # type: ignore[operator]

    def set_time(self, time: T) -> None:
        """Set the current time and update the time index."""
        # check time + dt is valid
        try:
            _ = time + self.dt  # type: ignore[operator]
        except Exception as e:
            raise TypeError(f"time must be compatible with dt={self.dt!r}") from e

        self._tidx = self.get_time_index(time)
        self._time = time

    def set_iteration(self, iteration: int) -> None:
        """Set the current iteration."""
        if iteration < 0:
            raise ValueError("Iteration must be non-negative.")
        self._iteration = iteration

    @property
    def time_unit(self) -> D:
        """The time unit for this clock."""
        return self._time_unit

    @property
    def dt(self) -> D:
        """The time step for this clock."""
        return (self._normalised_dt * self._time_unit).astype(self._time_unit.dtype)  # type: ignore[operator]

    @property
    def time(self) -> T:
        """The current time for this clock."""
        return self._time

    @property
    def time_array(self) -> npt.NDArray[T]:
        """The time array for this clock."""
        return self._time_array

    @property
    def iteration(self) -> int:
        """The current iteration for this clock."""
        return self._iteration

    @property
    def tidx(self) -> np.float64:
        """The current time index for this clock."""
        return self._tidx

    @property
    def tinfo(self) -> Tinfo:
        """The current time information for this clock."""
        return Time_info(self._time, self._tidx, self._iteration)

    @property
    def forward_in_time(self) -> bool:
        """Whether the clock is advancing time forwards."""
        return self._forward_in_time

    @property
    def first_time(self) -> T:
        """The chronological start of the simulation.

        Returns:
            T: ``time_array[0]`` if forward, ``time_array[-1]`` if backward.

        Examples:
            >>> clock = Clock(np.array([0.0, 1.0, 2.0, 3.0]), dt=np.float64(0.5))
            >>> clock.first_time
            np.float64(0.0)

            >>> clock = Clock(np.array([0.0, 1.0, 2.0, 3.0]), dt=np.float64(-0.5))
            >>> clock.first_time
            np.float64(3.0)
        """
        return self._time_array[0] if self._forward_in_time else self._time_array[-1]

    @property
    def final_time(self) -> T:
        """The chronological end of the simulation.

        Returns:
            T: ``time_array[-1]`` if forward, ``time_array[0]`` if backward.

        Examples:
            >>> clock = Clock(np.array([0.0, 1.0, 2.0, 3.0]), dt=np.float64(0.5))
            >>> clock.final_time
            np.float64(3.0)

            >>> clock = Clock(np.array([0.0, 1.0, 2.0, 3.0]), dt=np.float64(-0.5))
            >>> clock.final_time
            np.float64(0.0)
        """
        return self._time_array[-1] if self._forward_in_time else self._time_array[0]

    def advance_time(self) -> None:
        """Advance the current time by dt and update the time index."""
        self._time += self.dt  # type: ignore[operator]
        self._tidx = self.get_time_index(self._time)
        self._iteration += 1


class Timestepper(abc.ABC):
    """Class that handles particle timestepping."""

    def __init__(self) -> None:
        # default value for index padding
        self._index_padding = 0

        # initialise empty lists for kernels
        self._initialisation_kernels = []
        self._pre_step_kernels = []
        self._post_step_kernels = []

    def add_initialisation_kernels(self, *kernels: BoundKernel) -> None:
        """Add kernels to be launched during initialisation."""
        self._initialisation_kernels.extend(kernels)

    def add_pre_step_kernels(self, *kernels: BoundKernel) -> None:
        """Add kernels to be launched before each timestep."""
        self._pre_step_kernels.extend(kernels)

    def add_post_step_kernels(self, *kernels: BoundKernel) -> None:
        """Add kernels to be launched after each timestep."""
        self._post_step_kernels.extend(kernels)

    @property
    def index_padding(self) -> int:
        """The index padding required by this timestepper."""
        return self._index_padding

    @property
    def initialisation_kernels(self) -> list[BoundKernel]:
        """The initialisation kernels used by this timestepper."""
        return self._initialisation_kernels

    @property
    def pre_step_kernels(self) -> list[BoundKernel]:
        """The pre-step kernels used by this timestepper."""
        return self._pre_step_kernels

    @property
    def post_step_kernels(self) -> list[BoundKernel]:
        """The post-step kernels used by this timestepper."""
        return self._post_step_kernels

    @property
    def kernels(self) -> Iterator[BoundKernel]:
        """Get the kernels used by this timestepper."""
        return itertools.chain(
            self._initialisation_kernels,
            self._pre_step_kernels,
            self._post_step_kernels,
        )

    def run_initialisation(self, particles: Particles, launcher: Launcher, clock: Clock) -> None:
        """Initialize the particles by launching the initialisation kernels."""
        for kernel in self._initialisation_kernels:
            launcher.launch_kernel(kernel, particles, clock.tinfo)

    def run_pre_step(self, particles: Particles, launcher: Launcher, clock: Clock) -> None:
        """Launch the pre-step kernels."""
        for kernel in self._pre_step_kernels:
            launcher.launch_kernel(kernel, particles, clock.tinfo)

    @abc.abstractmethod
    def run_step(self, particles: Particles, launcher: Launcher, clock: Clock) -> None:
        """Timestep the particles."""
        pass

    def run_post_step(self, particles: Particles, launcher: Launcher, clock: Clock) -> None:
        """Launch the post-step kernels."""
        for kernel in self._post_step_kernels:
            launcher.launch_kernel(kernel, particles, clock.tinfo)


class RK2Timestepper(Timestepper):
    """Timestepper implements RK2 particle kernels.

    Implements two-stage second order explicit Runge-Kutta integration for particle advection.
    Explicit second-order RK2 schemes are defined by a single parameter alpha and have Butcher tableau:
        0   |
      alpha |       alpha
    -----------------------------------------
            |  1 - 1 / 2 alpha    1 / 2 alpha
    """

    def __init__(
        self,
        *,
        alpha: float = 2 / 3,
        index_padding: int = 0,
    ) -> None:
        super().__init__()
        self._rk_step_1_kernels = []
        self._rk_step_2_kernels = []
        self._rk_update_kernels = []
        self._alpha = alpha
        self._index_padding = index_padding

    def add_rk_step_1_kernels(self, *kernels: BoundKernel) -> None:
        """Add kernel to be launched during first rk step."""
        self._rk_step_1_kernels.extend(kernels)

    def add_rk_step_2_kernels(self, *kernels: BoundKernel) -> None:
        """Add kernel to be launched during second rk step."""
        self._rk_step_2_kernels.extend(kernels)

    def add_rk_update_kernels(self, *kernels: BoundKernel) -> None:
        """Add kernel to be launched during rk update step."""
        self._rk_update_kernels.extend(kernels)

    @property
    def alpha(self) -> float:
        """The RK2 alpha parameter used by this launcher."""
        return self._alpha

    @property
    def kernels(self) -> Iterator[BoundKernel]:
        """Get the kernels used by this timestepper."""
        return itertools.chain(
            super().kernels,
            self._rk_step_1_kernels,
            self._rk_step_2_kernels,
            self._rk_update_kernels,
        )

    def run_step(self, particles: Particles, launcher: Launcher, clock: Clock) -> None:
        """Launch the RK2 kernels to timestep the particles."""
        # Stage 1 - runs at current time
        for kernel in self._rk_step_1_kernels:
            launcher.launch_kernel(kernel, particles, clock.tinfo)
        # Compute intermediate time info
        intermediate_time = clock.time + self._alpha * clock.dt  # type: ignore[operator]
        intermediate_tidx = clock.get_time_index(intermediate_time)
        intermediate_tinfo = Time_info(intermediate_time, intermediate_tidx, clock.iteration)
        # Stage 2 - runs at intermediate time
        for kernel in self._rk_step_2_kernels:
            launcher.launch_kernel(kernel, particles, intermediate_tinfo)
        # Compute end time info
        end_time = clock.time + clock.dt  # type: ignore[operator]
        end_tidx = clock.get_time_index(end_time)
        end_tinfo = Time_info(end_time, end_tidx, clock.iteration)
        # Update kernel - runs at end time
        for kernel in self._rk_update_kernels:
            launcher.launch_kernel(kernel, particles, end_tinfo)


class ABTimestepper(Timestepper):
    """Class for Adams-Bashforth timesteppers."""

    def __init__(
        self,
        order: int = 3,
        *,
        index_padding: int = 0,
    ) -> None:
        super().__init__()
        self._tendency_kernels = []
        self._ab_update_kernels = []
        self._bump_status_kernel = construct_ab_bump_status_kernel()
        self._index_padding = index_padding
        self._order = order

        # validate order
        if self._order not in (2, 3):
            raise ValueError("Only orders 2 and 3 are currently supported for ABTimestepper.")

        # Add initialisation kernel
        self.add_initialisation_kernels(construct_ab_initialisation_kernel(order))

    def add_tendency_kernels(self, *kernels: BoundKernel) -> None:
        """Add kernels to be launched during tendency accumulation."""
        self._tendency_kernels.extend(kernels)

    def add_ab_update_kernels(self, *kernels: BoundKernel) -> None:
        """Add kernels to be launched during Adams-Bashforth update step."""
        self._ab_update_kernels.extend(kernels)

    def add_prognostic_property_kernel(
        self, prop: str, dprop_0: str | None = None, dprop_1: str | None = None, dprop_2: str | None = None
    ) -> None:
        """Create and add an Adams-Bashforth update kernel for the specified prognostic property.

        Args:
            prop: The name of the prognostic property.
            dprop_0: The name of the first derivative of the prognostic property. If None, defaults to prop + "_d0".
            dprop_1: The name of the second derivative of the prognostic property. If None, defaults to prop + "_d1".
            dprop_2: The name of the third derivative of the prognostic property. If None, defaults to prop + "_d2".
        """
        dprop_0 = dprop_0 or f"{prop}_d0"
        dprop_1 = dprop_1 or f"{prop}_d1"
        dprop_2 = dprop_2 or f"{prop}_d2"

        if self._order == 2:
            kernel = construct_ab2_update_kernel(prop, dprop_0, dprop_1)
        else:  # self._order == 3
            kernel = construct_ab3_update_kernel(prop, dprop_0, dprop_1, dprop_2)

        self.add_ab_update_kernels(kernel)

    @property
    def kernels(self) -> Iterator[BoundKernel]:
        """Get the kernels used by this timestepper."""
        return itertools.chain(
            super().kernels, self._tendency_kernels, self._ab_update_kernels, [self._bump_status_kernel]
        )

    def run_step(self, particles: Particles, launcher: Launcher, clock: Clock) -> None:
        """Launch the Adams-Bashforth kernel to timestep the particles."""

        # first launch tendency kernels to compute tendencies and store in dprop_0
        for kernel in self._tendency_kernels:
            launcher.launch_kernel(kernel, particles, clock.tinfo)
        # then the Adams-Bashforth update kernels
        for kernel in self._ab_update_kernels:
            launcher.launch_kernel(kernel, particles, clock.tinfo)
        # finally bump the particle status
        launcher.launch_kernel(self._bump_status_kernel, particles, clock.tinfo)
