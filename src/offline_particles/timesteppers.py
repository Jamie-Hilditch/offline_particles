"""Submodule for timestepping classes."""

import abc

import numpy as np
import numpy.typing as npt

from .kernels import ParticleKernel
from .launcher import Launcher, register_scalar_data_source


class Timestepper(abc.ABC):
    """Class that handles particle advection timestepping."""

    def __init__(
        self,
        time_array: npt.NDArray,
        dt: float,
        time: float = 0.0,
        iteration: int = 0,
        index_padding: int = 0,
    ) -> None:
        super().__init__()

        # check time array is strictly increasing
        if not np.all(np.diff(time_array) > 0):
            raise ValueError("Time array must be strictly increasing.")
        self._time_array = time_array

        # store iteration, timestep, current time and current time index
        self._iteration = iteration
        self._dt = dt
        self.set_time(time)

        self._index_padding = index_padding

    @property
    def iteration(self) -> int:
        """The current iteration for this timestepper."""
        return self._iteration

    @property
    def dt(self) -> float:
        """The time step for this timestepper."""
        return self._dt

    @property
    def time(self) -> float:
        """The current time for this timestepper."""
        return self._time

    @property
    def tidx(self) -> float:
        """The current time index for this timestepper."""
        return self._tidx

    @property
    def index_padding(self) -> int:
        """The index padding required by this timestepper."""
        return self._index_padding

    @register_scalar_data_source("_dt")
    def dt_scalar_data(self, *args) -> float:
        """get dt."""
        return self._dt

    @register_scalar_data_source("_time")
    def time_kernel_data(self, *args) -> float:
        """get time."""
        return self._time

    @register_scalar_data_source("_tidx")
    def tidx_kernel_data(self, *args) -> float:
        """get time index as field data."""
        return self._tidx

    def get_time_index(self, time: float) -> float:
        """Get the time index corresponding to the given time."""
        time_array = self._time_array
        if time < time_array[0] or time > time_array[-1]:
            raise ValueError("Time is out of bounds of the time array.")

        idx = np.searchsorted(time_array, time, side="right") - 1
        t0 = time_array[idx]
        t1 = time_array[idx + 1]
        fraction = (time - t0) / (t1 - t0)
        return idx + fraction

    def advance_time(self) -> None:
        """Advance the current time by dt and update the time index."""
        self._time += self._dt
        self._tidx = self.get_time_index(self._time)
        self._iteration += 1

    def set_time(self, time: float) -> None:
        """Set the current time and update the time index."""
        self._time = time
        self._tidx = self.get_time_index(time)

    @abc.abstractmethod
    def timestep_particles(self, particles: npt.NDArray, launcher: Launcher) -> None:
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

    def __init__(
        self,
        time_array: npt.NDArray,
        rk_step_1_kernel: ParticleKernel,
        rk_step_2_kernel: ParticleKernel,
        rk_update_kernel: ParticleKernel,
        dt: float,
        time: float = 0.0,
        *,
        alpha: float = 2 / 3,
        index_padding: int = 0,
        pre_step_kernel: ParticleKernel | None = None,
        post_step_kernel: ParticleKernel | None = None,
    ) -> None:
        super().__init__(time_array, dt, time, index_padding)
        self._rk_step_1_kernel = rk_step_1_kernel
        self._rk_step_2_kernel = rk_step_2_kernel
        self._rk_update_kernel = rk_update_kernel
        self._pre_step_kernel = pre_step_kernel
        self._post_step_kernel = post_step_kernel
        self._alpha = alpha

    @property
    def dt(self) -> float:
        """The time step used by this launcher."""
        return self._dt

    @property
    def alpha(self) -> float:
        """The RK2 alpha parameter used by this launcher."""
        return self._alpha

    @register_scalar_data_source("_RK2_alpha")
    def get_alpha_kernel_data(self, *args) -> float:
        """get alpha."""
        return self._alpha

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

    def timestep_particles(self, particles: npt.NDArray, launcher: Launcher) -> None:
        """Launch the RK2 kernels to timestep the particles."""
        if self._pre_step_kernel is not None:
            launcher.launch_kernel(self._pre_step_kernel, particles, self._tidx)
        # Stage 1
        launcher.launch_kernel(self._rk_step_1_kernel, particles, self._tidx)
        # Compute intermediate time and time index
        intermediate_time = self._time + self._alpha * self._dt
        intermediate_tidx = self.get_time_index(intermediate_time)
        # Stage 2
        launcher.launch_kernel(self._rk_step_2_kernel, particles, intermediate_tidx)
        # Advance time
        self.advance_time()
        # Update particle positions
        launcher.launch_kernel(self._rk_update_kernel, particles, self._tidx)
        if self._post_step_kernel is not None:
            launcher.launch_kernel(self._post_step_kernel, particles, self._tidx)
