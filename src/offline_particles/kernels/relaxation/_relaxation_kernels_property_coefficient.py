"""Kernels implementing relaxation forcing with a particle property relaxation coefficient."""

from typing import Callable, Literal

import numpy as np
import numpy.typing as npt

from ...spatial_arrays import ArrayAxis, ArrayLayout
from .._kernels import (
    BoundKernel,
    FieldDataDeclaration,
    FieldDataType,
    KernelFunction,
    ParticleKernel,
    ParticlePropertiesType,
    ParticlePropertyDeclaration,
    ScalarDeclaration,
    ScalarsType,
)
from ..input_declarations import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from ..layout_validators import ordering_validator_factory
from ._relaxation_impl import (
    _linear_relaxation_property_coefficient_1D_field_target,
    _linear_relaxation_property_coefficient_2D_field_target,
    _linear_relaxation_property_coefficient_3D_field_target,
    _linear_relaxation_property_coefficient_constant_target,
    _linear_relaxation_property_coefficient_property_target,
    _linear_relaxation_property_coefficient_scalar_target,
    _quadratic_relaxation_property_coefficient_1D_field_target,
    _quadratic_relaxation_property_coefficient_2D_field_target,
    _quadratic_relaxation_property_coefficient_3D_field_target,
    _quadratic_relaxation_property_coefficient_constant_target,
    _quadratic_relaxation_property_coefficient_property_target,
    _quadratic_relaxation_property_coefficient_scalar_target,
)

__all__ = [
    "construct_relaxation_kernel_property_coefficient_constant_target",
    "construct_relaxation_kernel_property_coefficient_property_target",
    "construct_relaxation_kernel_property_coefficient_scalar_target",
    "construct_relaxation_kernel_property_coefficient_field_target",
]

_INDEX_DECLARATION_MAPPING = {
    ArrayAxis.X: XIDX_DECLARATION,
    ArrayAxis.Y: YIDX_DECLARATION,
    ArrayAxis.Z: ZIDX_DECLARATION,
}


def construct_relaxation_kernel_property_coefficient_constant_target(
    prop: str,
    dprop: str,
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
    dprop : str
        The binding of the particle property accumulating tendency terms.
    relaxation_coefficient : str
        The binding of the property coefficient for the relaxation forcing.
    target : np.inexact | float
        The target value for the relaxation forcing.
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
    dtype = np.dtype(dtype)
    dtype_constructor = dtype.type

    if form == "linear":
        kernel_function_impl = _linear_relaxation_property_coefficient_constant_target(dtype_constructor(target))
    elif form == "quadratic":
        kernel_function_impl = _quadratic_relaxation_property_coefficient_constant_target(dtype_constructor(target))
    else:
        raise ValueError(f"Invalid form: {form}. Must be 'linear' or 'quadratic'.")

    def kernel_function(
        particle_properties: ParticlePropertiesType, scalars: ScalarsType, field_data: FieldDataType
    ) -> None:
        return kernel_function_impl(
            particle_properties["status"],
            particle_properties["prop"],
            particle_properties["relaxation_coefficient"],
            particle_properties["dprop"],
        )

    kernel = ParticleKernel(
        kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("prop", dtype),
            ParticlePropertyDeclaration("relaxation_coefficient", dtype),
            ParticlePropertyDeclaration("dprop", dtype),
        ],
    )

    bound_kernel = BoundKernel(
        kernel,
        particle_property_bindings={
            "prop": prop,
            "relaxation_coefficient": relaxation_coefficient,
            "dprop": dprop,
        },
    )

    return bound_kernel


def construct_relaxation_kernel_property_coefficient_property_target(
    prop: str,
    dprop: str,
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
    dprop : str
        The binding of the particle property accumulating tendency terms.
    relaxation_coefficient : str
        The binding of the particle property coefficient for the relaxation forcing.
    target : str
        The binding of the particle property containing the target value for the relaxation forcing.
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
    dtype = np.dtype(dtype)

    if form == "linear":
        kernel_function_impl = _linear_relaxation_property_coefficient_property_target()
    elif form == "quadratic":
        kernel_function_impl = _quadratic_relaxation_property_coefficient_property_target()
    else:
        raise ValueError(f"Invalid form: {form}. Must be 'linear' or 'quadratic'.")

    def kernel_function(
        particle_properties: ParticlePropertiesType, scalars: ScalarsType, field_data: FieldDataType
    ) -> None:
        return kernel_function_impl(
            particle_properties["status"],
            particle_properties["prop"],
            particle_properties["relaxation_coefficient"],
            particle_properties["target"],
            particle_properties["dprop"],
        )

    kernel = ParticleKernel(
        kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("prop", dtype),
            ParticlePropertyDeclaration("relaxation_coefficient", dtype),
            ParticlePropertyDeclaration("dprop", dtype),
            ParticlePropertyDeclaration("target", dtype),
        ],
    )

    bound_kernel = BoundKernel(
        kernel,
        particle_property_bindings={
            "prop": prop,
            "dprop": dprop,
            "relaxation_coefficient": relaxation_coefficient,
            "target": target,
        },
    )

    return bound_kernel


def construct_relaxation_kernel_property_coefficient_scalar_target(
    prop: str,
    dprop: str,
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
    dprop : str
        The binding of the particle property accumulating tendency terms.
    relaxation_coefficient : str
        The binding of the particle property containing the relaxation coefficient.
    target : str
        The binding of the scalar field containing the target value for the relaxation forcing.
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
    dtype = np.dtype(dtype)

    if form == "linear":
        kernel_function_impl = _linear_relaxation_property_coefficient_scalar_target()
    elif form == "quadratic":
        kernel_function_impl = _quadratic_relaxation_property_coefficient_scalar_target()
    else:
        raise ValueError(f"Invalid form: {form}. Must be 'linear' or 'quadratic'.")

    def kernel_function(
        particle_properties: ParticlePropertiesType, scalars: ScalarsType, field_data: FieldDataType
    ) -> None:
        return kernel_function_impl(
            particle_properties["status"],
            particle_properties["prop"],
            particle_properties["relaxation_coefficient"],
            particle_properties["dprop"],
            scalars["target"],
        )

    kernel = ParticleKernel(
        kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("prop", dtype),
            ParticlePropertyDeclaration("relaxation_coefficient", dtype),
            ParticlePropertyDeclaration("dprop", dtype),
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
            "dprop": dprop,
        },
        scalar_bindings={
            "target": target,
        },
    )

    return bound_kernel


def _construct_1d_kernel_function(
    kernel_function_impl: Callable[..., None],
    axis0: ArrayAxis,
) -> KernelFunction:
    """Construct a kernel function for a 1D array layout."""
    index_declaration = _INDEX_DECLARATION_MAPPING[axis0]
    idx0 = index_declaration.name

    def kernel_function(
        particle_properties: ParticlePropertiesType, scalars: ScalarsType, field_data: FieldDataType
    ) -> None:
        fd = field_data["target"]
        field_array = fd.array
        (offset,) = fd.offsets

        kernel_function_impl(
            particle_properties["status"],
            particle_properties[idx0],
            particle_properties["prop"],
            particle_properties["relaxation_coefficient"],
            particle_properties["dprop"],
            field_array,
            offset,
        )

    return kernel_function


def _construct_2d_kernel_function(
    kernel_function_impl: Callable[..., None],
    axis0: ArrayAxis,
    axis1: ArrayAxis,
) -> KernelFunction:
    """Construct a kernel function for a 2D array layout."""
    index_declaration_0 = _INDEX_DECLARATION_MAPPING[axis0]
    index_declaration_1 = _INDEX_DECLARATION_MAPPING[axis1]
    idx0 = index_declaration_0.name
    idx1 = index_declaration_1.name

    def kernel_function(
        particle_properties: ParticlePropertiesType, scalars: ScalarsType, field_data: FieldDataType
    ) -> None:
        fd = field_data["target"]
        field_array = fd.array
        offset_0, offset_1 = fd.offsets

        kernel_function_impl(
            particle_properties["status"],
            particle_properties[idx0],
            particle_properties[idx1],
            particle_properties["prop"],
            particle_properties["relaxation_coefficient"],
            particle_properties["dprop"],
            field_array,
            offset_0,
            offset_1,
        )

    return kernel_function


def _construct_3d_kernel_function(
    kernel_function_impl: Callable[..., None],
    axis0: ArrayAxis,
    axis1: ArrayAxis,
    axis2: ArrayAxis,
) -> KernelFunction:
    """Construct a kernel function for a 3D array layout."""
    index_declaration_0 = _INDEX_DECLARATION_MAPPING[axis0]
    index_declaration_1 = _INDEX_DECLARATION_MAPPING[axis1]
    index_declaration_2 = _INDEX_DECLARATION_MAPPING[axis2]
    idx0 = index_declaration_0.name
    idx1 = index_declaration_1.name
    idx2 = index_declaration_2.name

    def kernel_function(
        particle_properties: ParticlePropertiesType, scalars: ScalarsType, field_data: FieldDataType
    ) -> None:
        fd = field_data["target"]
        field_array = fd.array
        offset_0, offset_1, offset_2 = fd.offsets

        kernel_function_impl(
            particle_properties["status"],
            particle_properties[idx0],
            particle_properties[idx1],
            particle_properties[idx2],
            particle_properties["prop"],
            particle_properties["relaxation_coefficient"],
            particle_properties["dprop"],
            field_array,
            offset_0,
            offset_1,
            offset_2,
        )

    return kernel_function


def construct_relaxation_kernel_property_coefficient_field_target(
    prop: str,
    dprop: str,
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
    dprop : str
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
    """
    dtype = np.dtype(dtype)

    match form, array_layout.axes:
        # 1D field target
        case "linear", (axis0,):
            kernel_function_impl = _linear_relaxation_property_coefficient_1D_field_target(interpolation_half_width)
            index_declarations = [_INDEX_DECLARATION_MAPPING[axis0]]
            kernel_function = _construct_1d_kernel_function(kernel_function_impl, axis0)
        case "quadratic", (axis0,):
            kernel_function_impl = _quadratic_relaxation_property_coefficient_1D_field_target(interpolation_half_width)
            index_declarations = [_INDEX_DECLARATION_MAPPING[axis0]]
            kernel_function = _construct_1d_kernel_function(kernel_function_impl, axis0)
        # 2D field target
        case "linear", (axis0, axis1):
            kernel_function_impl = _linear_relaxation_property_coefficient_2D_field_target(interpolation_half_width)
            index_declarations = [_INDEX_DECLARATION_MAPPING[axis0], _INDEX_DECLARATION_MAPPING[axis1]]
            kernel_function = _construct_2d_kernel_function(kernel_function_impl, axis0, axis1)
        case "quadratic", (axis0, axis1):
            kernel_function_impl = _quadratic_relaxation_property_coefficient_2D_field_target(interpolation_half_width)
            index_declarations = [_INDEX_DECLARATION_MAPPING[axis0], _INDEX_DECLARATION_MAPPING[axis1]]
            kernel_function = _construct_2d_kernel_function(kernel_function_impl, axis0, axis1)
        # 3D field target
        case "linear", (axis0, axis1, axis2):
            kernel_function_impl = _linear_relaxation_property_coefficient_3D_field_target(interpolation_half_width)
            index_declarations = [
                _INDEX_DECLARATION_MAPPING[axis0],
                _INDEX_DECLARATION_MAPPING[axis1],
                _INDEX_DECLARATION_MAPPING[axis2],
            ]
            kernel_function = _construct_3d_kernel_function(kernel_function_impl, axis0, axis1, axis2)
        case "quadratic", (axis0, axis1, axis2):
            kernel_function_impl = _quadratic_relaxation_property_coefficient_3D_field_target(interpolation_half_width)
            index_declarations = [
                _INDEX_DECLARATION_MAPPING[axis0],
                _INDEX_DECLARATION_MAPPING[axis1],
                _INDEX_DECLARATION_MAPPING[axis2],
            ]
            kernel_function = _construct_3d_kernel_function(kernel_function_impl, axis0, axis1, axis2)
        case _:
            raise ValueError(f"Invalid array layout axes {array_layout.axes}.")

    validator = ordering_validator_factory(array_layout.axes)

    kernel = ParticleKernel(
        kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            *index_declarations,
            ParticlePropertyDeclaration("prop", dtype),
            ParticlePropertyDeclaration("relaxation_coefficient", dtype),
            ParticlePropertyDeclaration("dprop", dtype),
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
            "dprop": dprop,
        },
        field_data_bindings={
            "target": target,
        },
    )

    return bound_kernel
