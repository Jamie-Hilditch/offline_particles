"""Kernels for applying relaxation and damping to particle properties."""

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
]


def construct_linear_relaxation_kernel(
    prop: str,
    dprop: str,
    dtype: npt.DTypeLike = np.float64,
    *,
    constant_coefficient: np.inexact | None = None,
    property_coefficient: str | None = None,
    scalar_coefficient: str | None = None,
    constant_target: np.inexact | None = None,
    property_target: str | None = None,
    scalar_target: str | None = None,
    field_target: str | None = None,
    array_layout: ArrayLayout | None = None,
) -> BoundKernel:
    """Construct a kernel for applying linear relaxation to a particle property.

    Parameters
    ----------
    prop : str
        The binding for the particle property to relax.
    dprop : str
        The binding for the particle property to store the rate of change of `prop`.
    dtype : npt.DTypeLike, optional
        The data type of the particle properties, coefficient and target values, by default np.float64.
    constant_coefficient : np.inexact | None, optional
        A constant coefficient for the relaxation, by default None.
    property_coefficient : str | None, optional
        The binding for a particle property to use as the relaxation coefficient, by default None.
    scalar_coefficient : str | None, optional
        The binding for a scalar field to use as the relaxation coefficient, by default None.
    constant_target : np.inexact | None, optional
        A constant target value for the relaxation, by default None.
    property_target : str | None, optional
        The binding for a particle property to use as the target value for the relaxation, by default None.
    scalar_target : str | None, optional
        The binding for a scalar field to use as the target value for the relaxation, by default None.
    field_target : str | None, optional
        The binding for a field to use as the target value for the relaxation, by default None.
    array_layout : ArrayLayout | None, optional
        The layout of the particle properties in memory, by default None.

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
    `array_layout` must be provided if and only if `field_target` is specified.

    Linear relaxation is defined as:
    ```math
    d prop / d t = - coefficient * (prop - target)
    ```
    where `coefficient` is the relaxation coefficient, `target` is the target value for the property, and `prop` is the current value of the property.
    """
    coefficient = (constant_coefficient, property_coefficient, scalar_coefficient)
    target = (constant_target, property_target, scalar_target, field_target)

    if field_target is not None and array_layout is None:
        raise ValueError("`array_layout` must be provided when using a field target.")

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
            return construct_relaxation_kernel_constant_coefficient_field_target(
                prop, dprop, c, t, cast(ArrayLayout, array_layout), dtype
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
            return construct_relaxation_kernel_property_coefficient_field_target(
                prop, dprop, c, t, cast(ArrayLayout, array_layout), dtype
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
            return construct_relaxation_kernel_scalar_coefficient_field_target(
                prop, dprop, c, t, cast(ArrayLayout, array_layout), dtype
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
    constant_coefficient: np.inexact | None = None,
    property_coefficient: str | None = None,
    scalar_coefficient: str | None = None,
    constant_target: np.inexact | None = None,
    property_target: str | None = None,
    scalar_target: str | None = None,
    field_target: str | None = None,
    array_layout: ArrayLayout | None = None,
) -> BoundKernel:
    """Construct a kernel for applying quadratic relaxation to a particle property.

    Parameters
    ----------
    prop : str
        The binding for the particle property to relax.
    dprop : str
        The binding for the particle property to store the rate of change of `prop`.
    dtype : npt.DTypeLike, optional
        The data type of the particle properties, coefficient and target values, by default np.float64.
    constant_coefficient : np.inexact | None, optional
        A constant coefficient for the relaxation, by default None.
    property_coefficient : str | None, optional
        The binding for a particle property to use as the relaxation coefficient, by default None.
    scalar_coefficient : str | None, optional
        The binding for a scalar field to use as the relaxation coefficient, by default None.
    constant_target : np.inexact | None, optional
        A constant target value for the relaxation, by default None.
    property_target : str | None, optional
        The binding for a particle property to use as the target value for the relaxation, by default None.
    scalar_target : str | None, optional
        The binding for a scalar field to use as the target value for the relaxation, by default None.
    field_target : str | None, optional
        The binding for a field to use as the target value for the relaxation, by default None.
    array_layout : ArrayLayout | None, optional
        The layout of the particle properties in memory, by default None.

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
    `array_layout` must be provided if and only if `field_target` is specified.

    Quadratic relaxation is defined as:
    ```math
    d prop / d t = - coefficient * (prop - target) * |prop - target|
    ```
    where `coefficient` is the relaxation coefficient, `target` is the target value for the property, and `prop` is the current value of the property.
    """
    coefficient = (constant_coefficient, property_coefficient, scalar_coefficient)
    target = (constant_target, property_target, scalar_target, field_target)

    if field_target is not None and array_layout is None:
        raise ValueError("`array_layout` must be provided when using a field target.")

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
            return construct_relaxation_kernel_constant_coefficient_field_target(
                prop, dprop, c, t, cast(ArrayLayout, array_layout), dtype, form="quadratic"
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
            return construct_relaxation_kernel_property_coefficient_field_target(
                prop, dprop, c, t, cast(ArrayLayout, array_layout), dtype, form="quadratic"
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
            return construct_relaxation_kernel_scalar_coefficient_field_target(
                prop, dprop, c, t, cast(ArrayLayout, array_layout), dtype, form="quadratic"
            )

        case _:
            raise ValueError(
                "Exactly one coefficient (constant/property/scalar) and "
                "one target (constant/property/scalar/field) must be provided."
            )
