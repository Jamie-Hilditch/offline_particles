"""Kernels implementing relaxation forcing with a particle property relaxation coefficient."""

from typing import Literal

import numpy as np
import numpy.typing as npt

from ...spatial_arrays import ArrayAxis, ArrayLayout
from .._kernels import (
    BoundKernel,
    FieldDataDeclaration,
    ParticleKernel,
    ParticlePropertyDeclaration,
    ScalarDeclaration,
    kernel_function,
)
from ..input_declarations import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from ..layout_validators import ordering_validator_factory
from ._relaxation_impl import (
    _linear_relaxation_property_coefficient_1D_field_target_impl,
    _linear_relaxation_property_coefficient_2D_field_target_impl,
    _linear_relaxation_property_coefficient_3D_field_target_impl,
    _linear_relaxation_property_coefficient_constant_target,
    _linear_relaxation_property_coefficient_property_target,
    _linear_relaxation_property_coefficient_scalar_target,
    _quadratic_relaxation_property_coefficient_1D_field_target_impl,
    _quadratic_relaxation_property_coefficient_2D_field_target_impl,
    _quadratic_relaxation_property_coefficient_3D_field_target_impl,
    _quadratic_relaxation_property_coefficient_constant_target,
    _quadratic_relaxation_property_coefficient_property_target,
    _quadratic_relaxation_property_coefficient_scalar_target,
)

__all__ = [
    "construct_relaxation_kernel_property_coefficient_constant_target",
    "construct_relaxation_kernel_property_coefficient_field_target",
    "construct_relaxation_kernel_property_coefficient_property_target",
    "construct_relaxation_kernel_property_coefficient_scalar_target",
]

_INDEX_DECLARATION_MAPPING = {
    ArrayAxis.X: XIDX_DECLARATION,
    ArrayAxis.Y: YIDX_DECLARATION,
    ArrayAxis.Z: ZIDX_DECLARATION,
}


def construct_relaxation_kernel_property_coefficient_constant_target(
    prop: str,
    tendency: str,
    relaxation_coefficient: str,
    target: np.inexact | float,
    dtype: npt.DTypeLike = np.float64,
    form: Literal["linear", "quadratic"] = "linear",
) -> BoundKernel:
    """Construct a kernel to apply relaxation forcing to a particle property with property coefficient and constant target.

    Parameters
    ----------
    prop : str
        The binding of the particle property to which the relaxation forcing will be applied.
    tendency : str
        The binding of the particle property accumulating tendency terms.
    relaxation_coefficient : str
        The binding of the property coefficient for the relaxation forcing.
    target : np.inexact | float
        The target value for the relaxation forcing.
    dtype : npt.DTypeLike, optional
        The data type for the particle properties and constants used in the kernel. Default is np.float64.
    form : Literal["linear", "quadratic"], optional
        The form of the relaxation forcing, either "linear" or "quadratic". Default is "linear".

    Returns
    -------
    BoundKernel
        The constructed kernel for applying relaxation forcing to a particle property with property coefficient and constant target.

    Raises
    ------
    ValueError
        If the form is not "linear" or "quadratic".
    """
    dtype = np.dtype(dtype).type

    if form == "linear":
        kernel_fn = _linear_relaxation_property_coefficient_constant_target(dtype(target))
    elif form == "quadratic":
        kernel_fn = _quadratic_relaxation_property_coefficient_constant_target(dtype(target))
    else:
        raise ValueError(f"Invalid form: {form}. Must be 'linear' or 'quadratic'.")

    kernel = ParticleKernel(
        kernel_fn,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("tendency", dtype),
            ParticlePropertyDeclaration("prop", dtype),
            ParticlePropertyDeclaration("relaxation_coefficient", dtype),
        ],
    )

    bound_kernel = BoundKernel(
        kernel,
        particle_property_bindings={
            "prop": prop,
            "relaxation_coefficient": relaxation_coefficient,
            "tendency": tendency,
        },
    )

    return bound_kernel


def construct_relaxation_kernel_property_coefficient_property_target(
    prop: str,
    tendency: str,
    relaxation_coefficient: str,
    target: str,
    dtype: npt.DTypeLike = np.float64,
    form: Literal["linear", "quadratic"] = "linear",
) -> BoundKernel:
    """Construct a kernel to apply relaxation forcing to a particle property with property coefficient and property target.

    Parameters
    ----------
    prop : str
        The binding of the particle property to which the relaxation forcing will be applied.
    tendency : str
        The binding of the particle property accumulating tendency terms.
    relaxation_coefficient : str
        The binding of the particle property coefficient for the relaxation forcing.
    target : str
        The binding of the particle property containing the target value for the relaxation forcing.
    dtype : npt.DTypeLike, optional
        The data type for the particle properties and constants used in the kernel. Default is np.float64.
    form : Literal["linear", "quadratic"], optional
        The form of the relaxation forcing, either "linear" or "quadratic". Default is "linear".

    Returns
    -------
    BoundKernel
        The constructed kernel for applying relaxation forcing to a particle property with property coefficient and property target.

    Raises
    ------
    ValueError
        If the form is not "linear" or "quadratic".
    """
    dtype = np.dtype(dtype).type

    if form == "linear":
        kernel_fn = _linear_relaxation_property_coefficient_property_target()
    elif form == "quadratic":
        kernel_fn = _quadratic_relaxation_property_coefficient_property_target()
    else:
        raise ValueError(f"Invalid form: {form}. Must be 'linear' or 'quadratic'.")

    kernel = ParticleKernel(
        kernel_fn,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("tendency", dtype),
            ParticlePropertyDeclaration("prop", dtype),
            ParticlePropertyDeclaration("relaxation_coefficient", dtype),
            ParticlePropertyDeclaration("target", dtype),
        ],
    )

    bound_kernel = BoundKernel(
        kernel,
        particle_property_bindings={
            "prop": prop,
            "tendency": tendency,
            "relaxation_coefficient": relaxation_coefficient,
            "target": target,
        },
    )

    return bound_kernel


def construct_relaxation_kernel_property_coefficient_scalar_target(
    prop: str,
    tendency: str,
    relaxation_coefficient: str,
    target: str,
    dtype: npt.DTypeLike = np.float64,
    form: Literal["linear", "quadratic"] = "linear",
) -> BoundKernel:
    """Construct a kernel to apply relaxation forcing to a particle property with a property coefficient and scalar target.

    Parameters
    ----------
    prop : str
        The binding of the particle property to which the relaxation forcing will be applied.
    tendency : str
        The binding of the particle property accumulating tendency terms.
    relaxation_coefficient : str
        The binding of the particle property containing the relaxation coefficient.
    target : str
        The binding of the scalar field containing the target value for the relaxation forcing.
    dtype : npt.DTypeLike, optional
        The data type for the particle properties and constants used in the kernel. Default is np.float64.
    form : Literal["linear", "quadratic"], optional
        The form of the relaxation forcing, either "linear" or "quadratic". Default is "linear".

    Returns
    -------
    BoundKernel
        The constructed kernel for applying relaxation forcing to a particle property with property coefficient and scalar target.

    Raises
    ------
    ValueError
        If the form is not "linear" or "quadratic".
    """
    dtype = np.dtype(dtype).type

    if form == "linear":
        kernel_fn = _linear_relaxation_property_coefficient_scalar_target()
    elif form == "quadratic":
        kernel_fn = _quadratic_relaxation_property_coefficient_scalar_target()
    else:
        raise ValueError(f"Invalid form: {form}. Must be 'linear' or 'quadratic'.")

    kernel = ParticleKernel(
        kernel_fn,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("tendency", dtype),
            ParticlePropertyDeclaration("prop", dtype),
            ParticlePropertyDeclaration("relaxation_coefficient", dtype),
        ],
        scalars=[
            ScalarDeclaration("target", dtype),
        ],
    )

    bound_kernel = BoundKernel(
        kernel,
        particle_property_bindings={
            "prop": prop,
            "relaxation_coefficient": relaxation_coefficient,
            "tendency": tendency,
        },
        scalar_bindings={
            "target": target,
        },
    )

    return bound_kernel


def construct_relaxation_kernel_property_coefficient_field_target(
    prop: str,
    tendency: str,
    relaxation_coefficient: str,
    target: str,
    array_layout: ArrayLayout,
    dtype: npt.DTypeLike = np.float64,
    form: Literal["linear", "quadratic"] = "linear",
    interpolation_half_width: int = 1,
) -> BoundKernel:
    """Construct a kernel to apply relaxation forcing to a particle property with property coefficient and field target.

    Parameters
    ----------
    prop : str
        The binding of the particle property to which the relaxation forcing will be applied.
    tendency : str
        The binding of the particle property accumulating tendency terms.
    relaxation_coefficient : str
        The binding of the property coefficient for the relaxation forcing.
    target : str
        The binding of the field containing the target value for the relaxation forcing.
    array_layout : ArrayLayout
        The array layout of the field containing the target value for the relaxation forcing.
    dtype : npt.DTypeLike, optional
        The data type for the particle properties and scalars used in the kernel. Default is np.float64.
    form : Literal["linear", "quadratic"], optional
        The form of the relaxation forcing, either "linear" or "quadratic". Default is "linear".
    interpolation_half_width : int, optional
        The half-width of the interpolation stencil to use when interpolating the field target to the particle position.
        Default is 1, which corresponds to linear interpolation.

    Returns
    -------
    BoundKernel
        The constructed kernel for applying relaxation forcing to a particle property with property coefficient and field target.

    Raises
    ------
    ValueError
        If the form is not "linear" or "quadratic", or if the array layout is not supported.
    """
    dtype = np.dtype(dtype).type

    match form, array_layout.axes:
        # 1D field target
        case "linear", (_,):
            kernel_function_impl = _linear_relaxation_property_coefficient_1D_field_target_impl(
                interpolation_half_width
            )
        case "quadratic", (_,):
            kernel_function_impl = _quadratic_relaxation_property_coefficient_1D_field_target_impl(
                interpolation_half_width
            )
        # 2D field target
        case "linear", (_, _):
            kernel_function_impl = _linear_relaxation_property_coefficient_2D_field_target_impl(
                interpolation_half_width
            )
        case "quadratic", (_, _):
            kernel_function_impl = _quadratic_relaxation_property_coefficient_2D_field_target_impl(
                interpolation_half_width
            )
        # 3D field target
        case "linear", (_, _, _):
            kernel_function_impl = _linear_relaxation_property_coefficient_3D_field_target_impl(
                interpolation_half_width
            )
        case "quadratic", (_, _, _):
            kernel_function_impl = _quadratic_relaxation_property_coefficient_3D_field_target_impl(
                interpolation_half_width
            )
        case _:
            raise ValueError(f"Invalid array layout axes {array_layout.axes}.")

    index_declarations = [_INDEX_DECLARATION_MAPPING[axis] for axis in array_layout.axes]
    index_names = [decl.name for decl in index_declarations]
    validator = ordering_validator_factory(array_layout.axes)

    kernel_fn = kernel_function(
        particle_property_keys=["status", *index_names, "tendency", "prop", "relaxation_coefficient"],
        field_data_keys=["target"],
    )(kernel_function_impl)

    kernel = ParticleKernel(
        kernel_fn,
        particle_properties=[
            STATUS_DECLARATION,
            *index_declarations,
            ParticlePropertyDeclaration("tendency", dtype),
            ParticlePropertyDeclaration("prop", dtype),
            ParticlePropertyDeclaration("relaxation_coefficient", dtype),
        ],
        field_data=[
            FieldDataDeclaration("target", dtype, [validator]),
        ],
    )

    bound_kernel = BoundKernel(
        kernel,
        particle_property_bindings={
            "prop": prop,
            "relaxation_coefficient": relaxation_coefficient,
            "tendency": tendency,
        },
        field_data_bindings={
            "target": target,
        },
    )

    return bound_kernel
