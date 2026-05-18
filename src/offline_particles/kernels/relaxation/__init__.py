"""Kernels for applying relaxation and damping to particle properties."""

import warnings
from typing import cast

import numpy as np
import numpy.typing as npt

from ...spatial_arrays import ArrayLayout
from .._kernels import BoundKernel
from ._relaxation_kernels_constant_coefficient import (
    construct_relaxation_kernel_constant_coefficient_constant_target,
    construct_relaxation_kernel_constant_coefficient_field_target,
    construct_relaxation_kernel_constant_coefficient_property_target,
    construct_relaxation_kernel_constant_coefficient_scalar_target,
)
from ._relaxation_kernels_property_coefficient import (
    construct_relaxation_kernel_property_coefficient_constant_target,
    construct_relaxation_kernel_property_coefficient_field_target,
    construct_relaxation_kernel_property_coefficient_property_target,
    construct_relaxation_kernel_property_coefficient_scalar_target,
)
from ._relaxation_kernels_scalar_coefficient import (
    construct_relaxation_kernel_scalar_coefficient_constant_target,
    construct_relaxation_kernel_scalar_coefficient_field_target,
    construct_relaxation_kernel_scalar_coefficient_property_target,
    construct_relaxation_kernel_scalar_coefficient_scalar_target,
)

__all__ = [
    "construct_linear_relaxation_kernel",
    "construct_quadratic_relaxation_kernel",
    "construct_linear_damping_kernel",
    "construct_quadratic_damping_kernel",
]


def construct_linear_relaxation_kernel(
    prop: str,
    dprop: str,
    dtype: npt.DTypeLike = np.float64,
    *,
    constant_coefficient: np.inexact | float | None = None,
    property_coefficient: str | None = None,
    scalar_coefficient: str | None = None,
    constant_target: np.inexact | float | None = None,
    property_target: str | None = None,
    scalar_target: str | None = None,
    field_target: str | None = None,
    array_layout: ArrayLayout | None = None,
    interpolation_half_width: int | None = None,
) -> BoundKernel:
    r"""Construct a kernel for applying linear relaxation to a particle property.

    Parameters
    ----------
    prop : str
        The binding for the particle property to relax.
    dprop : str
        The binding for the particle property to store the rate of change of `prop`.
    dtype : npt.DTypeLike, optional
        The data type of the particle properties, coefficient and target values, by default np.float64.
    constant_coefficient : np.inexact | float | None, optional
        A constant coefficient for the relaxation, by default None.
    property_coefficient : str | None, optional
        The binding for a particle property to use as the relaxation coefficient, by default None.
    scalar_coefficient : str | None, optional
        The binding for a scalar field to use as the relaxation coefficient, by default None.
    constant_target : np.inexact | float | None, optional
        A constant target value for the relaxation, by default None.
    property_target : str | None, optional
        The binding for a particle property to use as the target value for the relaxation, by default None.
    scalar_target : str | None, optional
        The binding for a scalar field to use as the target value for the relaxation, by default None.
    field_target : str | None, optional
        The binding for a field to use as the target value for the relaxation, by default None.
    array_layout : ArrayLayout | None, optional
        The layout of the field target array, by default None.
    interpolation_half_width : int | None, optional
        The half-width of the interpolation stencil to use when interpolating the field target to particle positions.
        Defaults to 1 when `field_target` is specified, corresponding to linear interpolation. This parameter is only used when `field_target` is specified.

    Returns
    -------
    BoundKernel
        A kernel that applies linear relaxation to the specified particle property.

    Raises
    ------
    ValueError
        If zero or more than one of `constant_coefficient`, `property_coefficient`, or `scalar_coefficient` is provided,
        or if zero or more than one of `constant_target`, `property_target`, `scalar_target`, or `field_target` is provided,
        or if `array_layout` is not provided when `field_target` is specified.

    Notes
    -----
    Exactly one of `constant_coefficient`, `property_coefficient`, or `scalar_coefficient` must be provided, and
    exactly one of `constant_target`, `property_target`, `scalar_target`, or `field_target` must be provided.
    The kernel will apply the relaxation according to the specified coefficient and target.
    `array_layout` must be provided if `field_target` is specified.

    Linear relaxation is defined as:

    .. math::

        \frac{d\,\mathrm{prop}}{d\,t} = -\mathrm{coefficient} \left(\mathrm{prop} - \mathrm{target}\right)

    where `coefficient` is the relaxation coefficient, `target` is the target value for the property, and `prop` is the current value of the property.
    """
    coefficient = (constant_coefficient, property_coefficient, scalar_coefficient)
    target = (constant_target, property_target, scalar_target, field_target)

    if field_target is not None and array_layout is None:
        raise ValueError("`array_layout` must be provided when using a field target.")
    if field_target is None and array_layout is not None:
        warnings.warn(
            "`array_layout` is provided but `field_target` is None. `array_layout` will be ignored.",
            stacklevel=2,
        )
    if field_target is None and interpolation_half_width is not None:
        warnings.warn(
            "`interpolation_half_width` is provided but `field_target` is None. `interpolation_half_width` will be ignored.",
            stacklevel=2,
        )
    if field_target is not None and interpolation_half_width is None:
        interpolation_half_width = 1  # default to linear interpolation

    match coefficient, target:
        # constant coefficient
        case (c, None, None), (t, None, None, None) if c is not None and t is not None:
            return construct_relaxation_kernel_constant_coefficient_constant_target(prop, dprop, c, t, dtype)
        case (c, None, None), (None, t, None, None) if c is not None and t is not None:
            return construct_relaxation_kernel_constant_coefficient_property_target(prop, dprop, c, t, dtype)
        case (c, None, None), (None, None, t, None) if c is not None and t is not None:
            return construct_relaxation_kernel_constant_coefficient_scalar_target(prop, dprop, c, t, dtype)
        case (c, None, None), (None, None, None, t) if c is not None and t is not None:
            # cast array_layout to ArrayLayout to satisfy type checker, since we already checked that array_layout is not None
            # also cast interpolation_half_width to int to satisfy type checker, since we already checked that interpolation_half_width is not None
            return construct_relaxation_kernel_constant_coefficient_field_target(
                prop,
                dprop,
                c,
                t,
                cast(ArrayLayout, array_layout),
                dtype,
                interpolation_half_width=cast(int, interpolation_half_width),
            )

        # property coefficient
        case (None, c, None), (t, None, None, None) if c is not None and t is not None:
            return construct_relaxation_kernel_property_coefficient_constant_target(prop, dprop, c, t, dtype)
        case (None, c, None), (None, t, None, None) if c is not None and t is not None:
            return construct_relaxation_kernel_property_coefficient_property_target(prop, dprop, c, t, dtype)
        case (None, c, None), (None, None, t, None) if c is not None and t is not None:
            return construct_relaxation_kernel_property_coefficient_scalar_target(prop, dprop, c, t, dtype)
        case (None, c, None), (None, None, None, t) if c is not None and t is not None:
            # cast array_layout to ArrayLayout to satisfy type checker, since we already checked that array_layout is not None
            # also cast interpolation_half_width to int to satisfy type checker, since we already checked that interpolation_half_width is not None
            return construct_relaxation_kernel_property_coefficient_field_target(
                prop,
                dprop,
                c,
                t,
                cast(ArrayLayout, array_layout),
                dtype,
                interpolation_half_width=cast(int, interpolation_half_width),
            )

        # scalar coefficient
        case (None, None, c), (t, None, None, None) if c is not None and t is not None:
            return construct_relaxation_kernel_scalar_coefficient_constant_target(prop, dprop, c, t, dtype)
        case (None, None, c), (None, t, None, None) if c is not None and t is not None:
            return construct_relaxation_kernel_scalar_coefficient_property_target(prop, dprop, c, t, dtype)
        case (None, None, c), (None, None, t, None) if c is not None and t is not None:
            return construct_relaxation_kernel_scalar_coefficient_scalar_target(prop, dprop, c, t, dtype)
        case (None, None, c), (None, None, None, t) if c is not None and t is not None:
            # cast array_layout to ArrayLayout to satisfy type checker, since we already checked that array_layout is not None
            # also cast interpolation_half_width to int to satisfy type checker, since we already checked that interpolation_half_width is not None
            return construct_relaxation_kernel_scalar_coefficient_field_target(
                prop,
                dprop,
                c,
                t,
                cast(ArrayLayout, array_layout),
                dtype,
                interpolation_half_width=cast(int, interpolation_half_width),
            )

        case _:
            raise ValueError(
                "Exactly one coefficient (constant/property/scalar) and "
                "one target (constant/property/scalar/field) must be provided."
            )


def construct_quadratic_relaxation_kernel(
    prop: str,
    dprop: str,
    dtype: npt.DTypeLike = np.float64,
    *,
    constant_coefficient: np.inexact | float | None = None,
    property_coefficient: str | None = None,
    scalar_coefficient: str | None = None,
    constant_target: np.inexact | float | None = None,
    property_target: str | None = None,
    scalar_target: str | None = None,
    field_target: str | None = None,
    array_layout: ArrayLayout | None = None,
    interpolation_half_width: int | None = None,
) -> BoundKernel:
    r"""Construct a kernel for applying quadratic relaxation to a particle property.

    Parameters
    ----------
    prop : str
        The binding for the particle property to relax.
    dprop : str
        The binding for the particle property to store the rate of change of `prop`.
    dtype : npt.DTypeLike, optional
        The data type of the particle properties, coefficient and target values, by default np.float64.
    constant_coefficient : np.inexact | float | None, optional
        A constant coefficient for the relaxation, by default None.
    property_coefficient : str | None, optional
        The binding for a particle property to use as the relaxation coefficient, by default None.
    scalar_coefficient : str | None, optional
        The binding for a scalar field to use as the relaxation coefficient, by default None.
    constant_target : np.inexact | float | None, optional
        A constant target value for the relaxation, by default None.
    property_target : str | None, optional
        The binding for a particle property to use as the target value for the relaxation, by default None.
    scalar_target : str | None, optional
        The binding for a scalar field to use as the target value for the relaxation, by default None.
    field_target : str | None, optional
        The binding for a field to use as the target value for the relaxation, by default None.
    array_layout : ArrayLayout | None, optional
        The layout of the field target array, by default None.
    interpolation_half_width : int | None, optional
        The half-width of the interpolation stencil to use when interpolating the field target to particle positions.
        Defaults to 1 when `field_target` is specified, corresponding to linear interpolation. This parameter is only used when `field_target` is specified.

    Returns
    -------
    BoundKernel
        A kernel that applies quadratic relaxation to the specified particle property.

    Raises
    ------
    ValueError
        If zero or more than one of `constant_coefficient`, `property_coefficient`, or `scalar_coefficient` is provided,
        or if zero or more than one of `constant_target`, `property_target`, `scalar_target`, or `field_target` is provided,
        or if `array_layout` is not provided when `field_target` is specified.

    Notes
    -----
    Exactly one of `constant_coefficient`, `property_coefficient`, or `scalar_coefficient` must be provided, and
    exactly one of `constant_target`, `property_target`, `scalar_target`, or `field_target` must be provided.
    The kernel will apply the relaxation according to the specified coefficient and target.
    `array_layout` must be provided if `field_target` is specified.

    Quadratic relaxation is defined as:

    .. math::

        \frac{d\,\mathrm{prop}}{d\,t} = -\mathrm{coefficient} \left(\mathrm{prop} - \mathrm{target}\right) |\mathrm{prop} - \mathrm{target}|

    where `coefficient` is the relaxation coefficient, `target` is the target value for the property, and `prop` is the current value of the property.
    """
    coefficient = (constant_coefficient, property_coefficient, scalar_coefficient)
    target = (constant_target, property_target, scalar_target, field_target)

    if field_target is not None and array_layout is None:
        raise ValueError("`array_layout` must be provided when using a field target.")
    if field_target is None and array_layout is not None:
        warnings.warn(
            "`array_layout` is provided but `field_target` is None. `array_layout` will be ignored.",
            stacklevel=2,
        )
    if field_target is None and interpolation_half_width is not None:
        warnings.warn(
            "`interpolation_half_width` is provided but `field_target` is None. `interpolation_half_width` will be ignored.",
            stacklevel=2,
        )

    if field_target is not None and interpolation_half_width is None:
        interpolation_half_width = 1  # default to linear interpolation

    match coefficient, target:
        # constant coefficient
        case (c, None, None), (t, None, None, None) if c is not None and t is not None:
            return construct_relaxation_kernel_constant_coefficient_constant_target(
                prop, dprop, c, t, dtype, form="quadratic"
            )
        case (c, None, None), (None, t, None, None) if c is not None and t is not None:
            return construct_relaxation_kernel_constant_coefficient_property_target(
                prop, dprop, c, t, dtype, form="quadratic"
            )
        case (c, None, None), (None, None, t, None) if c is not None and t is not None:
            return construct_relaxation_kernel_constant_coefficient_scalar_target(
                prop, dprop, c, t, dtype, form="quadratic"
            )
        case (c, None, None), (None, None, None, t) if c is not None and t is not None:
            # cast array_layout to ArrayLayout to satisfy type checker, since we already checked that array_layout is not None
            # also cast interpolation_half_width to int to satisfy type checker, since we already checked that interpolation_half_width is not None
            return construct_relaxation_kernel_constant_coefficient_field_target(
                prop,
                dprop,
                c,
                t,
                cast(ArrayLayout, array_layout),
                dtype,
                interpolation_half_width=cast(int, interpolation_half_width),
                form="quadratic",
            )

        # property coefficient
        case (None, c, None), (t, None, None, None) if c is not None and t is not None:
            return construct_relaxation_kernel_property_coefficient_constant_target(
                prop, dprop, c, t, dtype, form="quadratic"
            )
        case (None, c, None), (None, t, None, None) if c is not None and t is not None:
            return construct_relaxation_kernel_property_coefficient_property_target(
                prop, dprop, c, t, dtype, form="quadratic"
            )
        case (None, c, None), (None, None, t, None) if c is not None and t is not None:
            return construct_relaxation_kernel_property_coefficient_scalar_target(
                prop, dprop, c, t, dtype, form="quadratic"
            )
        case (None, c, None), (None, None, None, t) if c is not None and t is not None:
            # cast array_layout to ArrayLayout to satisfy type checker, since we already checked that array_layout is not None
            # also cast interpolation_half_width to int to satisfy type checker, since we already checked that interpolation_half_width is not None
            return construct_relaxation_kernel_property_coefficient_field_target(
                prop,
                dprop,
                c,
                t,
                cast(ArrayLayout, array_layout),
                dtype,
                interpolation_half_width=cast(int, interpolation_half_width),
                form="quadratic",
            )

        # scalar coefficient
        case (None, None, c), (t, None, None, None) if c is not None and t is not None:
            return construct_relaxation_kernel_scalar_coefficient_constant_target(
                prop, dprop, c, t, dtype, form="quadratic"
            )
        case (None, None, c), (None, t, None, None) if c is not None and t is not None:
            return construct_relaxation_kernel_scalar_coefficient_property_target(
                prop, dprop, c, t, dtype, form="quadratic"
            )
        case (None, None, c), (None, None, t, None) if c is not None and t is not None:
            return construct_relaxation_kernel_scalar_coefficient_scalar_target(
                prop, dprop, c, t, dtype, form="quadratic"
            )
        case (None, None, c), (None, None, None, t) if c is not None and t is not None:
            # cast array_layout to ArrayLayout to satisfy type checker, since we already checked that array_layout is not None
            # also cast interpolation_half_width to int to satisfy type checker, since we already checked that interpolation_half_width is not None
            return construct_relaxation_kernel_scalar_coefficient_field_target(
                prop,
                dprop,
                c,
                t,
                cast(ArrayLayout, array_layout),
                dtype,
                interpolation_half_width=cast(int, interpolation_half_width),
                form="quadratic",
            )

        case _:
            raise ValueError(
                "Exactly one coefficient (constant/property/scalar) and "
                "one target (constant/property/scalar/field) must be provided."
            )


#################
# Special cases #
#################


def construct_linear_damping_kernel(
    prop: str,
    dprop: str,
    dtype: npt.DTypeLike = np.float64,
    *,
    constant_coefficient: np.inexact | float | None = None,
    property_coefficient: str | None = None,
    scalar_coefficient: str | None = None,
) -> BoundKernel:
    """Construct a kernel for applying linear damping to a particle property.

    Parameters
    ----------
    prop : str
        The binding for the particle property to relax.
    dprop : str
        The binding for the particle property to store the rate of change of `prop`.
    dtype : npt.DTypeLike, optional
        The data type of the particle properties, coefficient and target values, by default np.float64.
    constant_coefficient : np.inexact | float | None, optional
        A constant coefficient for the relaxation, by default None.
    property_coefficient : str | None, optional
        The binding for a particle property to use as the relaxation coefficient, by default None.
    scalar_coefficient : str | None, optional
        The binding for a scalar field to use as the relaxation coefficient, by default None.

    Returns
    -------
    BoundKernel
        A kernel that applies linear damping to the specified particle property.

    Raises
    ------
    ValueError
        If zero or more than one of `constant_coefficient`, `property_coefficient`, or `scalar_coefficient` is provided.

    Notes
    -----
    This is a special case of linear relaxation where the target value is zero.
    Exactly one of `constant_coefficient`, `property_coefficient`, or `scalar_coefficient` must be provided.
    The kernel will apply the damping according to the specified coefficient.

    Linear damping is defined as:
    ```math
    d prop / d t = - coefficient * prop
    ```
    where `coefficient` is the damping coefficient and `prop` is the current value of the property.
    """
    if sum(c is not None for c in (constant_coefficient, property_coefficient, scalar_coefficient)) != 1:
        raise ValueError("Exactly one coefficient (constant/property/scalar) must be provided.")

    return construct_linear_relaxation_kernel(
        prop,
        dprop,
        dtype=dtype,
        constant_coefficient=constant_coefficient,
        property_coefficient=property_coefficient,
        scalar_coefficient=scalar_coefficient,
        constant_target=0.0,
    )


def construct_quadratic_damping_kernel(
    prop: str,
    dprop: str,
    dtype: npt.DTypeLike = np.float64,
    *,
    constant_coefficient: np.inexact | float | None = None,
    property_coefficient: str | None = None,
    scalar_coefficient: str | None = None,
) -> BoundKernel:
    """Construct a kernel for applying quadratic damping to a particle property.

    Parameters
    ----------
    prop : str
        The binding for the particle property to relax.
    dprop : str
        The binding for the particle property to store the rate of change of `prop`.
    dtype : npt.DTypeLike, optional
        The data type of the particle properties, coefficient and target values, by default np.float64.
    constant_coefficient : np.inexact | float | None, optional
        A constant coefficient for the relaxation, by default None.
    property_coefficient : str | None, optional
        The binding for a particle property to use as the relaxation coefficient, by default None.
    scalar_coefficient : str | None, optional
        The binding for a scalar field to use as the relaxation coefficient, by default None.

    Returns
    -------
    BoundKernel
        A kernel that applies quadratic damping to the specified particle property.

    Raises
    ------
    ValueError
        If zero or more than one of `constant_coefficient`, `property_coefficient`, or `scalar_coefficient` is provided.

    Notes
    -----
    This is a special case of quadratic relaxation where the target value is zero.
    Exactly one of `constant_coefficient`, `property_coefficient`, or `scalar_coefficient` must be provided.
    The kernel will apply the damping according to the specified coefficient.

    Quadratic damping is defined as:

    .. math::

        d prop / d t = - coefficient * prop * |prop|

    where `coefficient` is the damping coefficient and `prop` is the current value of the property.
    """
    if sum(c is not None for c in (constant_coefficient, property_coefficient, scalar_coefficient)) != 1:
        raise ValueError("Exactly one coefficient (constant/property/scalar) must be provided.")

    return construct_quadratic_relaxation_kernel(
        prop,
        dprop,
        dtype=dtype,
        constant_coefficient=constant_coefficient,
        property_coefficient=property_coefficient,
        scalar_coefficient=scalar_coefficient,
        constant_target=0.0,
    )
