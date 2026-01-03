"""Submodule for timestepping classes."""

import abc

import numpy as np
import numpy.typing as npt

from .kernels import ParticleKernel
from .launcher import Launcher, ScalarSource
from .particles import Particles

type T = np.float64 | np.datetime64
type D = np.float64 | np.timedelta64


class Timestepper(abc.ABC):
    """Class that handles particle advection timestepping."""

    # scalar data sources
    _dt_scalar = ScalarSource("_dt", lambda self, tidx: self._normalised_dt)
    _time_scalar = ScalarSource("_time", lambda self, tidx: self.time)
    _tidx_scalar = ScalarSource("_tidx", lambda self, tidx: self._tidx)

    def __init__(
        self,
        time_array: npt.NDArray[T],
        dt: D,
        *,
        time_unit: D | None = None,
        time: T | None = None,
        iteration: int = 0,
        index_padding: int = 0,
    ) -> None:
        super().__init__()

        # check time_array is strictly increasing
        if np.any(time_array[1:] <= time_array[:-1]):  # type: ignore[operator]
            raise ValueError("time_array must be strictly increasing.")
        self._time_array = time_array

        # first set the time unit
        # this fixes the time types
        if time_unit is None:
            # use a default value of 1 if times are dimensionless else error
            if isinstance(dt, np.floating):
                time_unit = np.float64(1.0)
            else:
                raise ValueError("time_unit must be specified for dimensional time.")
        self._time_unit = time_unit

        # now set the timestep which has the same type as time_unit
        self.set_dt(dt)

        # store iteration, timestep, current time and current time index
        if time is None:
            time = self._time_array[0]
        self.set_time(time)
        self.set_iteration(iteration)

        # store index padding
        self._index_padding = 0
        self.set_index_padding(index_padding)

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
        t0 = time_array[idx]
        t1 = time_array[idx + 1]
        fraction = (time - t0) / (t1 - t0)
        return idx + fraction

    def set_dt(self, dt: D) -> None:
        """Set the time step for this timestepper."""
        # convert dt to timestep_type
        try:
            self._normalised_dt = np.float64(dt / self._time_unit)  # type: ignore[operator]
        except Exception as e:
            raise TypeError(f"dt must be of the same type as time_unit={self._time_unit!r}") from e

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
        """Set the current iteration for this timestepper."""
        if iteration < 0:
            raise ValueError("Iteration must be non-negative.")
        self._iteration = iteration

    def set_index_padding(self, index_padding: int, force: bool = False) -> None:
        """Set the index padding required by this timestepper.

        Unless `force` is True, only increases the index padding.
        """
        if index_padding < 0:
            raise ValueError("Index padding must be non-negative.")
        if force or index_padding > self._index_padding:
            self._index_padding = index_padding

    @property
    def time_unit(self) -> D:
        """The time unit for this timestepper."""
        return self._time_unit

    @property
    def dt(self) -> D:
        """The time step for this timestepper."""
        return self._normalised_dt * self._time_unit

    @property
    def time(self) -> T:
        """The current time for this timestepper."""
        return self._time

    @property
    def time_array(self) -> npt.NDArray[T]:
        """The time array for this timestepper."""
        return self._time_array

    @property
    def iteration(self) -> int:
        """The current iteration for this timestepper."""
        return self._iteration

    @property
    def tidx(self) -> np.float64:
        """The current time index for this timestepper."""
        return self._tidx

    @property
    def index_padding(self) -> int:
        """The index padding required by this timestepper."""
        return self._index_padding

    def advance_time(self) -> None:
        """Advance the current time by dt and update the time index."""
        self._time += self.dt  # type: ignore[operator]
        self._tidx = self.get_time_index(self._time)
        self._iteration += 1

    @property
    @abc.abstractmethod
    def kernels(self) -> tuple[ParticleKernel, ...]:
        """Get the kernels used by this timestepper."""
        pass

    @abc.abstractmethod
    def timestep_particles(self, particles: Particles, launcher: Launcher) -> None:
        """Timestep the particles by one time step."""
        pass


class RK2Timestepper(Timestepper):
    """Timestepper implements RK2 particle kernels.

    Implements two-stage second order explicit Runge-Kutta integration for particle advection.
    Explicit second-order RK2 schemes are defined by a single parameter alpha and have Butcher tableau:
        0   |
      alpha |       alpha
    -----------------------------------------
            |  1 - 1 / 2 alpha    1 / 2 alpha
    """

    # scalar source
    _alpha_scalar = ScalarSource("_RK2_alpha", lambda self, tidx: self._alpha)

    def __init__(
        self,
        time_array: npt.NDArray[T],
        dt: D,
        rk_step_1_kernel: ParticleKernel,
        rk_step_2_kernel: ParticleKernel,
        rk_update_kernel: ParticleKernel,
        *,
        alpha: float = 2 / 3,
        time_unit: D | None = None,
        time: T | None = None,
        iteration: int = 0,
        index_padding: int = 0,
        pre_step_kernel: ParticleKernel | None = None,
        post_step_kernel: ParticleKernel | None = None,
    ) -> None:
        super().__init__(
            time_array, dt, time_unit=time_unit, time=time, index_padding=index_padding, iteration=iteration
        )
        self._rk_step_1_kernel = rk_step_1_kernel
        self._rk_step_2_kernel = rk_step_2_kernel
        self._rk_update_kernel = rk_update_kernel
        self._pre_step_kernel = pre_step_kernel
        self._post_step_kernel = post_step_kernel
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        """The RK2 alpha parameter used by this launcher."""
        return self._alpha

    @property
    def kernels(self) -> tuple[ParticleKernel, ...]:
        """Get the kernels used by this timestepper."""
        if self._pre_step_kernel is not None:
            pre = (self._pre_step_kernel,)
        else:
            pre = ()
        if self._post_step_kernel is not None:
            post = (self._post_step_kernel,)
        else:
            post = ()
        return (
            pre
            + (
                self._rk_step_1_kernel,
                self._rk_step_2_kernel,
                self._rk_update_kernel,
            )
            + post
        )

    def timestep_particles(self, particles: Particles, launcher: Launcher) -> None:
        """Launch the RK2 kernels to timestep the particles."""
        if self._pre_step_kernel is not None:
            launcher.launch_kernel(self._pre_step_kernel, particles, self._tidx)
        # Stage 1
        launcher.launch_kernel(self._rk_step_1_kernel, particles, self._tidx)
        # Compute intermediate time and time index
        intermediate_time = self.time + self._alpha * self.dt  # type: ignore[operator]
        intermediate_tidx = self.get_time_index(intermediate_time)
        # Stage 2
        launcher.launch_kernel(self._rk_step_2_kernel, particles, intermediate_tidx)
        # Advance time
        self.advance_time()
        # Update particle positions
        launcher.launch_kernel(self._rk_update_kernel, particles, self._tidx)
        if self._post_step_kernel is not None:
            launcher.launch_kernel(self._post_step_kernel, particles, self._tidx)


class ABTimestepper(Timestepper):
    """Class for Adams-Bashforth timesteppers."""

    def __init__(
        self,
        time_array: npt.NDArray[T],
        dt: D,
        ab_kernel: ParticleKernel,
        *,
        time_unit: D | None = None,
        time: T | None = None,
        iteration: int = 0,
        index_padding: int = 0,
        pre_step_kernel: ParticleKernel | None = None,
        post_step_kernel: ParticleKernel | None = None,
    ) -> None:
        super().__init__(
            time_array, dt, time_unit=time_unit, time=time, index_padding=index_padding, iteration=iteration
        )
        self._ab_kernel = ab_kernel
        self._pre_step_kernel = pre_step_kernel
        self._post_step_kernel = post_step_kernel

    @property
    def kernels(self) -> tuple[ParticleKernel, ...]:
        """Get the kernels used by this timestepper."""
        if self._pre_step_kernel is not None:
            pre = (self._pre_step_kernel,)
        else:
            pre = ()
        if self._post_step_kernel is not None:
            post = (self._post_step_kernel,)
        else:
            post = ()
        return pre + (self._ab_kernel,) + post

    def timestep_particles(self, particles: Particles, launcher: Launcher) -> None:
        """Launch the Adams-Bashforth kernel to timestep the particles."""
        if self._pre_step_kernel is not None:
            launcher.launch_kernel(self._pre_step_kernel, particles, self._tidx)
        # Launch Adams-Bashforth kernel
        launcher.launch_kernel(self._ab_kernel, particles, self._tidx)
        # Advance time
        self.advance_time()
        if self._post_step_kernel is not None:
            launcher.launch_kernel(self._post_step_kernel, particles, self._tidx)
