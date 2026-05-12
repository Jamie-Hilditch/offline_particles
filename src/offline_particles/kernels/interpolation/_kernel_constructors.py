"""Constructors for 2N-point Lagrange interpolation kernels."""

import functools

import numpy as np
import numpy.typing as npt

from ...spatial_arrays import ArrayAxis
from .._kernels import (
    BoundKernel,
    FieldDataDeclaration,
    FieldDataType,
    ParticleKernel,
    ParticlePropertiesType,
    ParticlePropertyDeclaration,
    ScalarsType,
)
from ..input_declarations import STATUS_DECLARATION
from ..layout_validators import ordering_validator_factory
from ._lagrange import lagrange2N_1D_factory, lagrange2N_2D_factory, lagrange2N_3D_factory

__all__ = [
    "construct_1D_interpolation_kernel",
    "construct_2D_interpolation_kernel",
    "construct_3D_interpolation_kernel",
    "construct_Z_interpolation_kernel",
    "construct_Y_interpolation_kernel",
    "construct_X_interpolation_kernel",
    "construct_XY_interpolation_kernel",
    "construct_XZ_interpolation_kernel",
    "construct_YX_interpolation_kernel",
    "construct_YZ_interpolation_kernel",
    "construct_ZX_interpolation_kernel",
    "construct_ZY_interpolation_kernel",
    "construct_XYZ_interpolation_kernel",
    "construct_XZY_interpolation_kernel",
    "construct_YXZ_interpolation_kernel",
    "construct_YZX_interpolation_kernel",
    "construct_ZXY_interpolation_kernel",
    "construct_ZYX_interpolation_kernel",
]


def construct_1D_interpolation_kernel(
    axis: ArrayAxis | str,
    output: str,
    field: str,
    field_dtype: npt.DTypeLike = np.float64,
    output_dtype: npt.DTypeLike | None = None,
    N: int = 1,
    accumulate: bool = False,
) -> BoundKernel:
    """Construct a 1D interpolation kernel for the specified axis.

    Parameters
    ----------
    axis : ArrayAxis or str
        Axis to perform linear interpolation along. If a string is provided, it must be one of "Z", "Y", or "X".
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    field_dtype : npt.DTypeLike, optional
        Data type of the input field, by default np.float64.
    output_dtype : npt.DTypeLike | None, optional
        Data type of the output particle property, if None (default), field_dtype is used.
    N : int, optional
        Stencil size parameter for the Lagrange interpolation kernel. Must be a positive integer.
        Default is 1, which corresponds to linear interpolation. Higher values of N correspond to higher-order Lagrange polynomial interpolation.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.

    Returns
    -------
    BoundKernel
        A BoundKernel for performing interpolation along the specified axis.
    """
    axis = ArrayAxis.parse(axis)  # parse the axis argument to an ArrayAxis enum member
    idx_name = axis.particle_index_name
    validator = ordering_validator_factory((axis,))

    # get types for field and output
    field_dtype = np.dtype(field_dtype)
    if output_dtype is None:
        output_dtype = field_dtype
    else:
        output_dtype = np.dtype(output_dtype)

    # select kernel function implementation based on accumulate flag
    kernel_function_impl = lagrange2N_1D_factory(N=N, accumulate=accumulate)

    # wrap the kernel function implementation into the standard kernel function signature expected by ParticleKernel
    def kernel_function(
        particle_properties: ParticlePropertiesType,
        scalars: ScalarsType,
        fields: FieldDataType,
    ) -> None:
        field_data = fields["field"]
        kernel_function_impl(
            particle_properties["status"],
            particle_properties["idx"],
            particle_properties["output"],
            field_data.array,
            field_data.offsets[0],
        )

    # construct the ParticleKernel and then bind the particle properties and field data
    particle_kernel = ParticleKernel(
        kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("idx", np.float64),
            ParticlePropertyDeclaration("output", output_dtype),
        ],
        field_data=[
            FieldDataDeclaration("field", field_dtype, [validator]),
        ],
    )

    bound_kernel = BoundKernel(
        particle_kernel,
        particle_property_bindings={
            "idx": idx_name,
            "output": output,
        },
        field_data_bindings={
            "field": field,
        },
    )

    return bound_kernel


def construct_2D_interpolation_kernel(
    axes: tuple[ArrayAxis | str, ArrayAxis | str],
    output: str,
    field: str,
    field_dtype: npt.DTypeLike = np.float64,
    output_dtype: npt.DTypeLike | None = None,
    N: int = 1,
    accumulate: bool = False,
) -> BoundKernel:
    """Construct a 2D interpolation kernel for the specified axes.

    Parameters
    ----------
    axes : tuple[ArrayAxis or str, ArrayAxis or str]
        Tuple of two axes to perform bilinear interpolation along. If strings are provided, they must be one of "Z", "Y", or "X".
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    field_dtype : npt.DTypeLike, optional
        Data type of the input field, by default np.float64.
    output_dtype : npt.DTypeLike | None, optional
        Data type of the output particle property, if None (default) equal to the field_dtype.
    N : int, optional
        Stencil half width for the interpolation polynomial, by default 1 corresponding to linear interpolation.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.

    Returns
    -------
    BoundKernel
        A BoundKernel for performing 2D interpolation along the specified axes.
    """
    if len(axes) != 2:
        raise ValueError("axes must be a tuple of two elements.")

    axis_0 = ArrayAxis.parse(axes[0])
    axis_1 = ArrayAxis.parse(axes[1])

    if axis_0 == axis_1:
        raise ValueError(f"The two axes must be different. Received: {axis_0} and {axis_1}.")

    # get index names and validators for the specified axes
    idx_name_0 = axis_0.particle_index_name
    idx_name_1 = axis_1.particle_index_name
    validator = ordering_validator_factory((axis_0, axis_1))

    # get types for field and output
    field_dtype = np.dtype(field_dtype)
    if output_dtype is None:
        output_dtype = field_dtype
    else:
        output_dtype = np.dtype(output_dtype)

    # select kernel function implementation
    kernel_function_impl = lagrange2N_2D_factory(N=N, accumulate=accumulate)

    # wrap the kernel function implementation into the standard kernel function signature expected by ParticleKernel
    def kernel_function(
        particle_properties: ParticlePropertiesType,
        scalars: ScalarsType,
        fields: FieldDataType,
    ) -> None:
        field_data = fields["field"]
        kernel_function_impl(
            particle_properties["status"],
            particle_properties["idx_0"],
            particle_properties["idx_1"],
            particle_properties["output"],
            field_data.array,
            field_data.offsets[0],
            field_data.offsets[1],
        )

    # construct the ParticleKernel and then bind the particle properties and field data
    particle_kernel = ParticleKernel(
        kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("idx_0", np.float64),
            ParticlePropertyDeclaration("idx_1", np.float64),
            ParticlePropertyDeclaration("output", output_dtype),
        ],
        field_data=[
            FieldDataDeclaration("field", field_dtype, [validator]),
        ],
    )

    bound_kernel = BoundKernel(
        particle_kernel,
        particle_property_bindings={
            "idx_0": idx_name_0,
            "idx_1": idx_name_1,
            "output": output,
        },
        field_data_bindings={
            "field": field,
        },
    )

    return bound_kernel


def construct_3D_interpolation_kernel(
    axes: tuple[ArrayAxis | str, ArrayAxis | str, ArrayAxis | str],
    output: str,
    field: str,
    field_dtype: npt.DTypeLike = np.float64,
    output_dtype: npt.DTypeLike | None = None,
    N: int = 1,
    accumulate: bool = False,
) -> BoundKernel:
    """Construct a 3D interpolation kernel.

    Parameters
    ----------
    axes : tuple[ArrayAxis or str, ArrayAxis or str, ArrayAxis or str]
        Tuple of three axes to perform trilinear interpolation along. If strings are provided, they must be one of "Z", "Y", or "X".
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    field_dtype : npt.DTypeLike, optional
        Data type of the input field, by default np.float64.
    output_dtype : npt.DTypeLike | None, optional
        Data type of the output particle property, if None (default) equal to the field dtype.
    N : int, optional
        Stencil half width for the interpolation polynomial, by default 1 corresponding to linear interpolation.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.

    Returns
    -------
    BoundKernel
        A BoundKernel for performing trilinear interpolation.
    """
    if len(axes) != 3:
        raise ValueError("axes must be a tuple of three elements.")

    axis_0 = ArrayAxis.parse(axes[0])
    axis_1 = ArrayAxis.parse(axes[1])
    axis_2 = ArrayAxis.parse(axes[2])

    if len({axis_0, axis_1, axis_2}) != 3:
        raise ValueError(f"All three axes must be different. Received: {axis_0}, {axis_1}, and {axis_2}.")

    # get index names and validators for the specified axes
    idx_name_0 = axis_0.particle_index_name
    idx_name_1 = axis_1.particle_index_name
    idx_name_2 = axis_2.particle_index_name
    validator = ordering_validator_factory((axis_0, axis_1, axis_2))

    # get types for field and output
    field_dtype = np.dtype(field_dtype)
    if output_dtype is None:
        output_dtype = field_dtype
    else:
        output_dtype = np.dtype(output_dtype)

    # select kernel function implementation
    kernel_function_impl = lagrange2N_3D_factory(N=N, accumulate=accumulate)

    # wrap the kernel function implementation into the standard kernel function signature expected by ParticleKernel
    def kernel_function(
        particle_properties: ParticlePropertiesType,
        scalars: ScalarsType,
        fields: FieldDataType,
    ) -> None:
        field_data = fields["field"]
        kernel_function_impl(
            particle_properties["status"],
            particle_properties["idx_0"],
            particle_properties["idx_1"],
            particle_properties["idx_2"],
            particle_properties["output"],
            field_data.array,
            field_data.offsets[0],
            field_data.offsets[1],
            field_data.offsets[2],
        )

    # construct the ParticleKernel and then bind the particle properties and field data
    particle_kernel = ParticleKernel(
        kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("idx_0", np.float64),
            ParticlePropertyDeclaration("idx_1", np.float64),
            ParticlePropertyDeclaration("idx_2", np.float64),
            ParticlePropertyDeclaration("output", output_dtype),
        ],
        field_data=[
            FieldDataDeclaration("field", field_dtype, [validator]),
        ],
    )

    bound_kernel = BoundKernel(
        particle_kernel,
        particle_property_bindings={
            "idx_0": idx_name_0,
            "idx_1": idx_name_1,
            "idx_2": idx_name_2,
            "output": output,
        },
        field_data_bindings={
            "field": field,
        },
    )

    return bound_kernel


############################
# convenience constructors #
############################

#: Partial function application of :func:`construct_1D_interpolation_kernel` with axis=ArrayAxis.Z.
construct_Z_interpolation_kernel = functools.partial(construct_1D_interpolation_kernel, axis=ArrayAxis.Z)
#: Partial function application of :func:`construct_1D_interpolation_kernel` with axis=ArrayAxis.Y.
construct_Y_interpolation_kernel = functools.partial(construct_1D_interpolation_kernel, axis=ArrayAxis.Y)
#: Partial function application of :func:`construct_1D_interpolation_kernel` with axis=ArrayAxis.X.
construct_X_interpolation_kernel = functools.partial(construct_1D_interpolation_kernel, axis=ArrayAxis.X)

#: Partial function application of :func:`construct_2D_interpolation_kernel` with axes=(ArrayAxis.X, ArrayAxis.Y).
construct_XY_interpolation_kernel = functools.partial(
    construct_2D_interpolation_kernel, axes=(ArrayAxis.X, ArrayAxis.Y)
)
#: Partial function application of :func:`construct_2D_interpolation_kernel` with axes=(ArrayAxis.X, ArrayAxis.Z).
construct_XZ_interpolation_kernel = functools.partial(
    construct_2D_interpolation_kernel, axes=(ArrayAxis.X, ArrayAxis.Z)
)
#: Partial function application of :func:`construct_2D_interpolation_kernel` with axes=(ArrayAxis.Y, ArrayAxis.X).
construct_YX_interpolation_kernel = functools.partial(
    construct_2D_interpolation_kernel, axes=(ArrayAxis.Y, ArrayAxis.X)
)
#: Partial function application of :func:`construct_2D_interpolation_kernel` with axes=(ArrayAxis.Y, ArrayAxis.Z).
construct_YZ_interpolation_kernel = functools.partial(
    construct_2D_interpolation_kernel, axes=(ArrayAxis.Y, ArrayAxis.Z)
)
#: Partial function application of :func:`construct_2D_interpolation_kernel` with axes=(ArrayAxis.Z, ArrayAxis.X).
construct_ZX_interpolation_kernel = functools.partial(
    construct_2D_interpolation_kernel, axes=(ArrayAxis.Z, ArrayAxis.X)
)
#: Partial function application of :func:`construct_2D_interpolation_kernel` with axes=(ArrayAxis.Z, ArrayAxis.Y).
construct_ZY_interpolation_kernel = functools.partial(
    construct_2D_interpolation_kernel, axes=(ArrayAxis.Z, ArrayAxis.Y)
)

#: Partial function application of :func:`construct_3D_interpolation_kernel` with axes=(ArrayAxis.X, ArrayAxis.Y, ArrayAxis.Z).
construct_XYZ_interpolation_kernel = functools.partial(
    construct_3D_interpolation_kernel, axes=(ArrayAxis.X, ArrayAxis.Y, ArrayAxis.Z)
)
#: Partial function application of :func:`construct_3D_interpolation_kernel` with axes=(ArrayAxis.X, ArrayAxis.Z, ArrayAxis.Y).
construct_XZY_interpolation_kernel = functools.partial(
    construct_3D_interpolation_kernel, axes=(ArrayAxis.X, ArrayAxis.Z, ArrayAxis.Y)
)
#: Partial function application of :func:`construct_3D_interpolation_kernel` with axes=(ArrayAxis.Y, ArrayAxis.X, ArrayAxis.Z).
construct_YXZ_interpolation_kernel = functools.partial(
    construct_3D_interpolation_kernel, axes=(ArrayAxis.Y, ArrayAxis.X, ArrayAxis.Z)
)
#: Partial function application of :func:`construct_3D_interpolation_kernel` with axes=(ArrayAxis.Y, ArrayAxis.Z, ArrayAxis.X).
construct_YZX_interpolation_kernel = functools.partial(
    construct_3D_interpolation_kernel, axes=(ArrayAxis.Y, ArrayAxis.Z, ArrayAxis.X)
)
#: Partial function application of :func:`construct_3D_interpolation_kernel` with axes=(ArrayAxis.Z, ArrayAxis.X, ArrayAxis.Y).
construct_ZXY_interpolation_kernel = functools.partial(
    construct_3D_interpolation_kernel, axes=(ArrayAxis.Z, ArrayAxis.X, ArrayAxis.Y)
)
#: Partial function application of :func:`construct_3D_interpolation_kernel` with axes=(ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X).
construct_ZYX_interpolation_kernel = functools.partial(
    construct_3D_interpolation_kernel, axes=(ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X)
)
