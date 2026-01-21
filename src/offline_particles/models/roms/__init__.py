"""Offline particles simulations using ROMS output."""

import numpy as np

# import ROMS kernels
from ...kernels import ParticleKernel
from ...kernels.adams_bashforth import ab3_bump_status_kernel, ab3_update_kernel
from ...kernels.roms import (
    ab3_w_advection_kernel,
    compute_z_kernel,
    compute_zidx_kernel,
    rk2_w_advection_step_1_kernel,
    rk2_w_advection_step_2_kernel,
    rk2_w_advection_update_kernel,
    xyidx_tendency_linear_interpolation_kernel,
    z_tendency_buoyancy_driven_kernel,
    z_tendency_linearly_interpolate_w_kernel,
)
from ...kernels.validation import validation_kernel
from ...timestepping import ABTimestepper, RK2Timestepper

__all__ = [
    "compute_z_kernel",
    "ab3_w_advection_kernel",
    "ab3_post_w_advection_kernel",
    "roms_ab3_timestepper",
    "rk2_w_advection_step_1_kernel",
    "rk2_w_advection_step_2_kernel",
    "rk2_w_advection_update_kernel",
    "rk2_w_advection_timestepper",
]

type D = np.float64 | np.timedelta64

# create timesteppers for ROMS simulations with preset kernels


def rk2_w_advection_timestepper(
    *,
    index_padding: int = 5,
    alpha: float = 2 / 3,
) -> RK2Timestepper:
    """Create an RK2 timestepper with ROMS w advection kernels.

    Keyword Args:
        index_padding: Index padding, i.e. the minimum amount by which the field indices
            exceed the particle indices (default 5).
        alpha: The RK2 alpha parameter (default 2/3 - the Ralston method).
    """
    timestepper = RK2Timestepper(
        alpha=alpha,
        index_padding=index_padding,
    )
    timestepper.add_pre_step_kernels(validation_kernel)
    timestepper.add_rk_step_1_kernels(rk2_w_advection_step_1_kernel)
    timestepper.add_rk_step_2_kernels(rk2_w_advection_step_2_kernel)
    timestepper.add_rk_update_kernels(rk2_w_advection_update_kernel)
    timestepper.add_post_step_kernels(compute_zidx_kernel())
    return timestepper


def roms_ab3_timestepper(
    *,
    vertical_velocity: bool = True,
    buoyant_particles: bool = False,
    index_padding: int = 5,
) -> ABTimestepper:
    """Create an AB3 timestepper with ROMS advection kernels.

    Keyword Args:
        vertical_velocity: Whether to include vertical velocity advection (default True).
        buoyant_particles: Whether to include a buoyancy driven component to the vertical velocity (default False).
        time_unit: Unit of time for the simulation (default None). Should be the same type as dt.
        index_padding: Index padding, i.e. the minimum amount by which the field indices
            exceed the particle indices (default 5).

    Notes:
        ROMS uses a sigma coordinate system in the vertical. Vertical advection occurs in physical space,
        i.e. in `z`. Therefore, after each advection step, the particle `zidx` is recomputed based on the updated `z` position.

        !!! Important !!!
        Both `z` and `zidx` must be initialised before the simulation start. `roms.compute_z_kernel` can be used to compute `z` from `zidx`
        and `roms.compute_zidx_kernel` can be used to compute `zidx` from `z`.

        Horizontal advection occurs in index space, i.e. in `xidx` and `yidx`.

        Vertical advection using `w` (vertical velocity) is optional and can be disabled by setting `vertical_velocity=False`.
        The particles can be made buoyant by setting `buoyant_particles=True`, which adds a buoyancy driven component to the vertical velocity.
        This adds a "buoyancy velocity" `wb` to the particle. The tendency of `wb` is computed based on the local density difference between the
        particle (rho_p) and the surrounding fluid (rho).
            dwb/dt = (rho - rho_p) * g / rho0 - buoyancy_velocity_damping * wb

    """
    # construct the tendency kernels based on the options
    tendency_kernels = []

    # horizontal advection
    tendency_kernels.append(
        xyidx_tendency_linear_interpolation_kernel("_dxidx0", "_dyidx0"),
    )

    # vertical advection
    if vertical_velocity:
        tendency_kernels.append(z_tendency_linearly_interpolate_w_kernel("_dz0"))
    if buoyant_particles:
        tendency_kernels.append(z_tendency_buoyancy_driven_kernel("_dz0", "_dwb0"))

    # AB3 steps
    ab_kernels = []
    ab_kernels.append(ab3_update_kernel("xidx", "_dxidx0", "_dxidx1"))
    ab_kernels.append(ab3_update_kernel("yidx", "_dyidx0", "_dyidx1"))

    if vertical_velocity or buoyant_particles:
        ab_kernels.append(ab3_update_kernel("z", "_dz0", "_dz1"))
    if buoyant_particles:
        ab_kernels.append(ab3_update_kernel("wb", "_dwb0", "_dwb1"))

    # finally add the status bump kernel
    ab_kernels.append(ab3_bump_status_kernel)

    # combine kernels
    ab_kernel = ParticleKernel.chain(*tendency_kernels, *ab_kernels)

    # post step kernel to update zidx after advection
    post_step_kernels = [compute_zidx_kernel()]

    # pre step validation kernel
    pre_step_kernels = [validation_kernel]

    timestepper = ABTimestepper(
        index_padding=index_padding,
    )
    timestepper.add_pre_step_kernels(*pre_step_kernels)
    timestepper.add_ab_kernels(ab_kernel)
    timestepper.add_post_step_kernels(*post_step_kernels)
    return timestepper
