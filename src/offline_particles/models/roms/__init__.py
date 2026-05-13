"""Offline particles simulations using ROMS output."""

# import ROMS kernels
from ...kernels.base import construct_add_property_kernel
from ...kernels.buoyancy import construct_buoyancy_force_accumulation_kernel
from ...kernels.interpolation import construct_ZYX_interpolation_kernel
from ...kernels.relaxation import (
    construct_linear_damping_accumulation_kernel,
    construct_quadratic_damping_accumulation_kernel,
)
from ...kernels.roms import (
    construct_compute_z_kernel,
    construct_compute_zidx_kernel,
    construct_horizontal_idx_tendency_kernel_from_velocity_field,
)
from ...kernels.timestepping import construct_ab3_update_kernel, construct_ab_bump_status_kernel
from ...kernels.validation import construct_validation_kernel
from ...timestepping import ABTimestepper

__all__ = [
    "roms_ab3_timestepper",
    "construct_compute_zidx_kernel",
    "construct_compute_z_kernel",
]


def roms_ab3_timestepper(
    *,
    vertical_velocity: bool = True,
    buoyant_particles: bool = False,
    linear_damping: bool = False,
    quadratic_damping: bool = False,
    index_padding: int = 5,
    u: str = "u",
    v: str = "v",
    w: str = "w",
    dx: str = "dx",
    dy: str = "dy",
    h: str = "h",
    zeta: str = "zeta",
    C: str = "C",
    rho: str = "rho",
    hc: str = "hc",
    NZ: str = "NZ",
    g: str = "g",
    rho0: str = "rho0",
    linear_damping_coefficient: str = "linear_damping_coefficient",
    quadratic_damping_coefficient: str = "quadratic_damping_coefficient",
) -> ABTimestepper:
    """Create an AB3 timestepper with ROMS advection kernels.

    Keyword Args:
        vertical_velocity: Whether to include vertical velocity advection (default True).
        buoyant_particles: Whether to include a buoyancy driven component to the vertical velocity (default False).
        time_unit: Unit of time for the simulation (default None). Should be the same type as dt.
        index_padding: Index padding, i.e. the minimum amount by which the field indices
            exceed the particle indices (default 5).
        u: Binding for the u velocity field (default "u").
        v: Binding for the v velocity field (default "v").
        w: Binding for the w velocity field (default "w"). Only used if `vertical_velocity` is True.
        dx: Binding for the x grid spacing field (default "dx").
        dy: Binding for the y grid spacing field (default "dy").
        h: Binding for the bathymetry field (default "h").
        zeta: Binding for the free surface field (default "zeta").
        C: Binding for the vertical stretching function field (default "C").
        rho: Binding for the density field (default "rho"). Only used if `buoyant_particles` is True.
        hc: Binding for the critical depth scalar (default "hc").
        NZ: Binding for the number of vertical levels scalar (default "NZ").
        g: Binding for the gravitational acceleration scalar (default "g"). Only used if `buoyant_particles` is True.
        rho0: Binding for the reference density scalar (default "rho0"). Only used if `buoyant_particles` is True.
        linear_damping_coefficient: Binding for the linear damping coefficient scalar (default "linear_damping_coefficient").
            Only used if `linear_damping` is True.
        quadratic_damping_coefficient: Binding for the quadratic damping coefficient scalar (default "quadratic_damping_coefficient").
            Only used if `quadratic_damping` is True.

    Notes:
        ROMS uses a sigma coordinate system in the vertical. Vertical advection occurs in physical space,
        i.e. in `z`. Therefore, after each advection step, the particle `zidx` is recomputed based on the updated `z` position.

        !!! Important !!!
        Both `z` and `zidx` must be initialised before the simulation start. `roms.construct_compute_z_kernel()` can be used to
        construct a kernel to compute `z` from `zidx` and `roms.construct_compute_zidx_kernel` can be used to construct a kernel
        to compute `zidx` from `z`.

        Horizontal advection occurs in index space, i.e. in `xidx` and `yidx`.

        Vertical advection using `w` (vertical velocity) is optional and can be disabled by setting `vertical_velocity=False`.
        The particles can be made buoyant by setting `buoyant_particles=True`, which adds a buoyancy driven component to the
        vertical velocity. This adds a relative vertical velocity `w_rel` to the particle. The tendency of `w_rel` is computed
        based on the local density difference between the particle (rho_particle) and the surrounding fluid (rho_environment).
            dw_rel/dt = (rho_environment - rho_particle) * g / rho0
        where g is the gravitational acceleration and rho0 is a reference density. Damping can be applied to `w_rel` using linear
        and/or quadratic damping by setting `linear_damping=True` and/or `quadratic_damping=True`.
    """
    # construct the tendency kernels based on the options
    tendency_kernels = []

    # horizontal advection
    tendency_kernels.extend(
        [
            construct_horizontal_idx_tendency_kernel_from_velocity_field("_dxidx0", u, dx),
            construct_horizontal_idx_tendency_kernel_from_velocity_field("_dyidx0", v, dy),
        ]
    )

    # vertical advection
    if vertical_velocity:
        tendency_kernels.append(construct_ZYX_interpolation_kernel("_dz0", w, accumulate=True))

    # relative vertical velocity
    if buoyant_particles or linear_damping or quadratic_damping:
        tendency_kernels.append(construct_add_property_kernel("w_rel", "_dz0"))
    if buoyant_particles:
        tendency_kernels.append(
            construct_buoyancy_force_accumulation_kernel(
                "_dw_rel0", density_field=rho, reference_density=rho0, gravity=g
            )
        )
    if linear_damping:
        tendency_kernels.append(
            construct_linear_damping_accumulation_kernel("_dw_rel0", "w_rel", linear_damping_coefficient)
        )
    if quadratic_damping:
        tendency_kernels.append(
            construct_quadratic_damping_accumulation_kernel("_dw_rel0", "w_rel", quadratic_damping_coefficient)
        )

    # AB3 steps
    ab_kernels = []
    ab_kernels.append(construct_ab3_update_kernel("xidx", "_dxidx0", "_dxidx1", "_dxidx2"))
    ab_kernels.append(construct_ab3_update_kernel("yidx", "_dyidx0", "_dyidx1", "_dyidx2"))

    if vertical_velocity or buoyant_particles or linear_damping or quadratic_damping:
        ab_kernels.append(construct_ab3_update_kernel("z", "_dz0", "_dz1", "_dz2"))
    if buoyant_particles or linear_damping or quadratic_damping:
        ab_kernels.append(construct_ab3_update_kernel("w_rel", "_dw_rel0", "_dw_rel1", "_dw_rel2"))

    # finally add the status bump kernel
    ab_kernels.append(construct_ab_bump_status_kernel())

    # post step kernel to update zidx after advection
    post_step_kernels = [construct_compute_zidx_kernel()]

    # pre step validation kernel
    pre_step_kernels = [construct_validation_kernel()]

    timestepper = ABTimestepper(
        index_padding=index_padding,
    )
    timestepper.add_pre_step_kernels(*pre_step_kernels)
    timestepper.add_ab_kernels(*tendency_kernels, *ab_kernels)
    timestepper.add_post_step_kernels(*post_step_kernels)
    return timestepper
