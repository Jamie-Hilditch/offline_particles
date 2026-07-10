"""Kernels for working with buoyant particles."""

import numpy as np
import numpy.typing as npt

from ...spatial_arrays import ArrayLayout
from .._kernels import BoundKernel
from ..relaxation import construct_linear_relaxation_kernel


def construct_buoyancy_force_kernel(
    rhs: str,
    particle_density: str,
    density_field: str,
    array_layout: ArrayLayout,
    dtype: npt.DTypeLike = np.float64,
    *,
    constant_coefficient: np.inexact | float | None = None,
    property_coefficient: str | None = None,
    scalar_coefficient: str | None = None,
    interpolation_half_width: int | None = None,
) -> BoundKernel:
    r"""Construct a kernel to compute buoyancy force on particles.

    This is a special case of linear relaxation where the particle feels a restoring force
    towards the it's level of neutral buoyancy. For density based models, the coefficient
    is :math:`g/\rho_0`.

    Parameters
    ----------
    rhs : str
        The binding for the particle property to add the computed buoyancy force to.
    particle_density : str
        The binding for the particle property that contains the particle density.
    density_field : str
        The binding for the field data that contains the ambient fluid density.
    array_layout : ArrayLayout
        The layout of the density field array.
    dtype : npt.DTypeLike, optional
        The data type of the particle properties and coefficient, by default np.float64.
    constant_coefficient : np.inexact | float | None, optional
        A constant value for :math:`g/\rho_0`, by default None.
    property_coefficient : str | None, optional
        The binding for a particle property containing :math:`g/\rho_0`, by default None.
    scalar_coefficient : str | None, optional
        The binding for a scalar containing :math:`g/\rho_0`, by default None.
    interpolation_half_width : int | None, optional
        The half-width of the interpolation stencil to use when interpolating the ambient density field to
        particle positions. Defaults to 1, corresponding to trilinear interpolation.

    Returns
    -------
    BoundKernel
        A bound kernel that computes the buoyancy force on particles.

    Raises
    ------
    ValueError
        If zero or more than one of `constant_coefficient`, `property_coefficient`, or `scalar_coefficient` is provided.

    Notes
    -----
    Exactly one of `constant_coefficient`, `property_coefficient`, or `scalar_coefficient` must be provided.

    Buoyancy forcing is defined as:

    .. math::

        \frac{d\,\mathrm{rhs}}{d\,t} = -\frac{g}{\rho_0} \left(\rho_{\mathrm{particle}} - \rho_{\mathrm{env}}\right)

    where :math:`\rho_{\mathrm{particle}}` is `particle_density` and :math:`\rho_{\mathrm{env}}` is interpolated
    from `density_field`.
    """
    if sum(c is not None for c in (constant_coefficient, property_coefficient, scalar_coefficient)) != 1:
        raise ValueError("Exactly one coefficient (constant/property/scalar) must be provided.")

    return construct_linear_relaxation_kernel(
        rhs,
        particle_density,
        dtype=dtype,
        constant_coefficient=constant_coefficient,
        property_coefficient=property_coefficient,
        scalar_coefficient=scalar_coefficient,
        field_target=density_field,
        array_layout=array_layout,
        interpolation_half_width=interpolation_half_width,
    )
