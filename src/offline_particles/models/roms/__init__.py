"""Offline particles simulations using ROMS output."""

import numpy as np
import numpy.typing as npt

# import ROMS kernels
from ...kernels.roms import (
    ab3_post_w_advection_kernel,
    ab3_w_advection_kernel,
    compute_z_kernel,
    rk2_w_advection_step_1_kernel,
    rk2_w_advection_step_2_kernel,
    rk2_w_advection_update_kernel,
)
from ...kernels.validation import validation_kernel
from ...timesteppers import ABTimestepper, RK2Timestepper

__all__ = [
    "compute_z_kernel",
    "ab3_w_advection_kernel",
    "ab3_post_w_advection_kernel",
    "ab3_w_advection_timestepper",
    "rk2_w_advection_step_1_kernel",
    "rk2_w_advection_step_2_kernel",
    "rk2_w_advection_update_kernel",
    "rk2_w_advection_timestepper",
]

type D = np.float64 | np.timedelta64
type T = np.float64 | np.datetime64

# create timesteppers for ROMS simulations with preset kernels


def rk2_w_advection_timestepper(
    time_array: npt.NDArray,
    dt: D,
    *,
    time: T | None = None,
    time_unit: D | None = None,
    iteration: int = 0,
    index_padding: int = 5,
    alpha: float = 2 / 3,
) -> RK2Timestepper:
    """Create an RK2 timestepper with ROMS w advection kernels.

    Args:
        time_array: Array of simulation times.
        dt: Timestep size.

    Keyword Args:
        time: Initial simulation time.
        iteration: Initial iteration number.
        index_padding: Index padding, i.e. the minimum amount by which the field indices
            exceed the particle indices (default 5).
        alpha: The RK2 alpha parameter (default 2/3 - the Ralston method).
    """
    return RK2Timestepper(
        time_array,
        dt,
        rk_step_1_kernel=rk2_w_advection_step_1_kernel,
        rk_step_2_kernel=rk2_w_advection_step_2_kernel,
        rk_update_kernel=rk2_w_advection_update_kernel,
        time=time,
        time_unit=time_unit,
        iteration=iteration,
        index_padding=index_padding,
        alpha=alpha,
        pre_step_kernel=validation_kernel,
    )


def ab3_w_advection_timestepper(
    time_array: npt.NDArray,
    dt: D,
    *,
    time: T | None = None,
    time_unit: D | None = None,
    iteration: int = 0,
    index_padding: int = 5,
) -> ABTimestepper:
    """Create an AB3 timestepper with ROMS w advection kernels.

    Args:
        time_array: Array of simulation times.
        dt: Timestep size.

    Keyword Args:
        time: Initial simulation time.
        iteration: Initial iteration number.
        index_padding: Index padding, i.e. the minimum amount by which the field indices
            exceed the particle indices (default 5).
    """
    return ABTimestepper(
        time_array,
        dt,
        ab_kernel=ab3_w_advection_kernel,
        time=time,
        time_unit=time_unit,
        iteration=iteration,
        index_padding=index_padding,
        pre_step_kernel=validation_kernel,
        post_step_kernel=ab3_post_w_advection_kernel,
    )
