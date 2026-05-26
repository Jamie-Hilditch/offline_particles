"""Offline particles simulations using ROMS output."""

import numpy as np

from ...kernels.advection import construct_advection_kernel
from ...kernels.base import construct_add_property_kernel
from ...kernels.buoyancy import construct_buoyancy_force_accumulation_kernel
from ...kernels.interpolation import construct_ZYX_interpolation_kernel
from ...kernels.relaxation import (
    construct_linear_damping_kernel,
    construct_quadratic_damping_kernel,
)
from ...kernels.roms import (
    construct_compute_z_kernel,
    construct_compute_zidx_kernel,
)
from ...kernels.timestepping import construct_ab3_update_kernel
from ...kernels.validation import construct_validation_kernel
from ...spatial_arrays import BBox
from ...timestepping import ABTimestepper

__all__ = [
    "BBox",
    "construct_compute_z_kernel",
    "construct_compute_zidx_kernel",
    "construct_validation_kernel",
    "roms_ab3_timestepper",
]


def roms_ab3_timestepper(
    *,
    vertical_velocity: bool = True,
    buoyant_particles: bool = False,
    index_padding: int = 5,
    u: str = "u",
    v: str = "v",
    w: str = "w",
    pm: str = "pm",
    pn: str = "pn",
    h: str = "h",
    zeta: str = "zeta",
    C: str = "C",
    rho: str = "rho",
    hc: str = "hc",
    NZ: str = "NZ",
    g: str = "g",
    rho0: str = "rho0",
    constant_linear_damping_coefficient: np.inexact | float | None = None,
    constant_quadratic_damping_coefficient: np.inexact | float | None = None,
    property_linear_damping_coefficient: str | None = None,
    property_quadratic_damping_coefficient: str | None = None,
    scalar_linear_damping_coefficient: str | None = None,
    scalar_quadratic_damping_coefficient: str | None = None,
) -> ABTimestepper:
    r"""Create an AB3 timestepper with ROMS advection kernels.

    Parameters
    ----------
    vertical_velocity : bool, optional
        Whether to include vertical velocity advection (default True).
    buoyant_particles : bool, optional
        Whether to include a buoyancy driven component to the vertical velocity (default False).
    index_padding : int, optional
        Index padding, i.e. the minimum amount by which the field indices
        exceed the particle indices (default 5).
    u : str, optional
        Binding for the u velocity field (default "u").
    v : str, optional
        Binding for the v velocity field (default "v").
    w : str, optional
        Binding for the w velocity field (default "w"). Only used if `vertical_velocity` is True.
    pm : str, optional
        Binding for the pm (xi metric) field (default "pm").
    pn : str, optional
        Binding for the pn (eta metric) field (default "pn").
    h : str, optional
        Binding for the bathymetry field (default "h").
    zeta : str, optional
        Binding for the free surface field (default "zeta").
    C : str, optional
        Binding for the vertical stretching function field (default "C").
    rho : str, optional
        Binding for the density field (default "rho"). Only used if `buoyant_particles` is True.
    hc : str, optional
        Binding for the critical depth scalar (default "hc").
    NZ : str, optional
        Binding for the number of vertical levels scalar (default "NZ").
    g : str, optional
        Binding for the gravitational acceleration scalar (default "g"). Only used if `buoyant_particles` is True.
    rho0 : str, optional
        Binding for the reference density scalar (default "rho0"). Only used if `buoyant_particles` is True.
    constant_linear_damping_coefficient : np.inexact | float | None, optional
        If not None, include linear damping with a constant damping coefficient (default None).
    constant_quadratic_damping_coefficient : np.inexact | float | None, optional
        If not None, include quadratic damping with a constant damping coefficient (default None).
    property_linear_damping_coefficient : str, optional
        If provided the binding for the particle property to use as the linear damping coefficient.
    property_quadratic_damping_coefficient : str, optional
        If provided the binding for the particle property to use as the quadratic damping coefficient.
    scalar_linear_damping_coefficient : str, optional
        If provided the binding for a scalar field to use as the linear damping coefficient.
    scalar_quadratic_damping_coefficient : str, optional
        If provided the binding for a scalar field to use as the quadratic damping coefficient.

    Returns
    -------
    ABTimestepper
        Timestepper with ROMS advection kernels.

    Raises
    ------
    ValueError
        If more than one of the linear or quadratic damping coefficient arguments are provided.
        From :function:`construct_linear_damping_kernel` or :function:`construct_quadratic_damping_kernel`.

    Notes
    -----
    ROMS uses a sigma coordinate system in the vertical. Vertical advection occurs in physical space,
    i.e. in :math:`z`. Therefore, after each advection step, the particle `zidx` is recomputed based on the updated :math:`z` position.

    .. warning::

        Both `z` and `zidx` must be initialised before the simulation start. `roms.construct_compute_z_kernel()` can be used to
        construct a kernel to compute `z` from `zidx` and `roms.construct_compute_zidx_kernel` can be used to construct a kernel
        to compute `zidx` from `z`.

    Horizontal advection occurs in index space, i.e. in `xidx` and `yidx`.

    Vertical advection using `w` (vertical velocity) is optional and can be disabled by setting `vertical_velocity=False`.
    The particles can be made buoyant by setting `buoyant_particles=True`, which adds a buoyancy driven component to the
    vertical velocity. This adds a relative vertical velocity `w_rel` to the particle. The tendency of `w_rel` is computed
    based on the local density difference between the particle :math:`(\rho_{\mathrm{particle}})` and the surrounding fluid :math:`(\rho_{\mathrm{env}})`.

    .. math::

        \frac{dw_{\mathrm{rel}}}{dt} = \frac{(\rho_{\mathrm{env}} - \rho_{\mathrm{particle}})}{\rho_0}g

    where :math:`g` is the gravitational acceleration and :math:`\rho_0` is a reference density.

    Damping can be applied to `w_rel` using linear damping by specifying at most one of
    `constant_linear_damping_coefficient`, `property_linear_damping_coefficient`, `scalar_linear_damping_coefficient`.
    Similar for quadratic damping at most one of `constant_quadratic_damping_coefficient`, `property_quadratic_damping_coefficient`,
    `scalar_quadratic_damping_coefficient` may be provided.
    These arguments are passed onto :function:`construct_linear_damping_kernel` and :function:`construct_quadratic_damping_kernel`.
    """
    # construct the tendency kernels based on the options
    tendency_kernels = []

    # horizontal advection
    x_advection_kernel = construct_advection_kernel("_dxidx0", u, pm, ("Z", "Y", "X"), ("Y", "X"), metric=True)
    y_advection_kernel = construct_advection_kernel("_dyidx0", v, pn, ("Z", "Y", "X"), ("Y", "X"), metric=True)
    tendency_kernels.extend([x_advection_kernel, y_advection_kernel])

    # vertical advection
    if vertical_velocity:
        tendency_kernels.append(construct_ZYX_interpolation_kernel("_dz0", w, accumulate=True))

    # see if we're adding linear or quadratic damping to the relative vertical velocity
    linear_damping = (
        constant_linear_damping_coefficient is not None
        or property_linear_damping_coefficient is not None
        or scalar_linear_damping_coefficient is not None
    )
    quadratic_damping = (
        constant_quadratic_damping_coefficient is not None
        or property_quadratic_damping_coefficient is not None
        or scalar_quadratic_damping_coefficient is not None
    )

    # relative vertical velocity damping
    if linear_damping:
        linear_damping_kernel = construct_linear_damping_kernel(
            "w_rel",
            "_dw_rel0",
            constant_coefficient=constant_linear_damping_coefficient,
            property_coefficient=property_linear_damping_coefficient,
            scalar_coefficient=scalar_linear_damping_coefficient,
        )
        tendency_kernels.append(linear_damping_kernel)
    if quadratic_damping:
        quadratic_damping_kernel = construct_quadratic_damping_kernel(
            "w_rel",
            "_dw_rel0",
            constant_coefficient=constant_quadratic_damping_coefficient,
            property_coefficient=property_quadratic_damping_coefficient,
            scalar_coefficient=scalar_quadratic_damping_coefficient,
        )
        tendency_kernels.append(quadratic_damping_kernel)

    # buoyancy forcing
    if buoyant_particles:
        tendency_kernels.append(
            construct_buoyancy_force_accumulation_kernel(
                "_dw_rel0", density_field=rho, reference_density=rho0, gravity=g
            )
        )

    # if we're including buoyancy or damping we need to add to the tendency for the vertical position
    if buoyant_particles or linear_damping or quadratic_damping:
        tendency_kernels.append(construct_add_property_kernel("w_rel", "_dz0"))

    # AB3 steps
    ab_kernels = []
    ab_kernels.append(construct_ab3_update_kernel("xidx", "_dxidx0", "_dxidx1", "_dxidx2"))
    ab_kernels.append(construct_ab3_update_kernel("yidx", "_dyidx0", "_dyidx1", "_dyidx2"))

    if vertical_velocity or buoyant_particles or linear_damping or quadratic_damping:
        ab_kernels.append(construct_ab3_update_kernel("z", "_dz0", "_dz1", "_dz2"))
    if buoyant_particles or linear_damping or quadratic_damping:
        ab_kernels.append(construct_ab3_update_kernel("w_rel", "_dw_rel0", "_dw_rel1", "_dw_rel2"))

    # post step kernel to update zidx after advection
    post_step_kernels = [construct_compute_zidx_kernel(hc=hc, NZ=NZ, h=h, zeta=zeta, C=C)]

    timestepper = ABTimestepper(
        index_padding=index_padding,
        order=3,
    )

    timestepper.add_tendency_kernels(*tendency_kernels)
    timestepper.add_ab_update_kernels(*ab_kernels)
    timestepper.add_post_step_kernels(*post_step_kernels)
    return timestepper
