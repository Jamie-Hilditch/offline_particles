"""Constructors for Lagrange polynomial interpolation kernels."""

import functools

import numpy as np
import numpy.typing as npt

from ...spatial_arrays import ArrayAxis
from .._kernels import (
    BoundKernel,
    FieldDataDeclaration,
    ParticleKernel,
    ParticlePropertyDeclaration,
)
from ..input_declarations import STATUS_DECLARATION
from ..layout_validators import ordering_validator_factory
from ._lagrange_kernel_functions import (
    _lagrange_1D_kernel_function_factory,
    _lagrange_2D_kernel_function_factory,
    _lagrange_3D_kernel_function_factory,
)

__all__ = [
    "construct_1D_interpolation_kernel",
    "construct_2D_interpolation_kernel",
    "construct_3D_interpolation_kernel",
    "construct_XYZ_interpolation_kernel",
    "construct_XY_interpolation_kernel",
    "construct_XZY_interpolation_kernel",
    "construct_XZ_interpolation_kernel",
    "construct_X_interpolation_kernel",
    "construct_YXZ_interpolation_kernel",
    "construct_YX_interpolation_kernel",
    "construct_YZX_interpolation_kernel",
    "construct_YZ_interpolation_kernel",
    "construct_Y_interpolation_kernel",
    "construct_ZXY_interpolation_kernel",
    "construct_ZX_interpolation_kernel",
    "construct_ZYX_interpolation_kernel",
    "construct_ZY_interpolation_kernel",
    "construct_Z_interpolation_kernel",
]


def construct_1D_interpolation_kernel(
    axis: ArrayAxis | str,
    output: str,
    field: str,
    field_dtype: npt.DTypeLike = np.float64,
    output_dtype: npt.DTypeLike | None = None,
    N: int = 1,
    D: int = 0,
    accumulate: bool = False,
) -> BoundKernel:
    """Construct a 1D interpolation kernel for the specified axis.

    Parameters
    ----------
    axis : ArrayAxis or str
        Axis to perform interpolation along. If a string is provided, it must be one of "Z", "Y", or "X".
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    field_dtype : npt.DTypeLike, optional
        Data type of the input field, by default np.float64.
    output_dtype : npt.DTypeLike | None, optional
        Data type of the output particle property, if `None` (default), `field_dtype` is used.
    N : int, optional
        Stencil size parameter for the Lagrange interpolation kernel. Must be a positive integer.
        Default is 1, which corresponds to linear interpolation. Higher values of `N` correspond to higher-order Lagrange polynomial interpolation.
    D : int, optional
        Derivative order of the Lagrange interpolation kernel. Must be a non-negative integer.
        Default is 0, which corresponds to value interpolation. Higher values correspond to derivatives with respect to index space.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.

    Returns
    -------
    BoundKernel
        A BoundKernel for performing interpolation along the specified axis.

    Raises
    ------
    ValueError
        If `axis` is not a valid :class:`ArrayAxis`.
        If `N` is not a positive integer.
        If `D` is not a non-negative integer.
    """
    axis = ArrayAxis.parse(axis)  # parse the axis argument to an ArrayAxis enum member
    idx_name = axis.particle_index_name
    validator = ordering_validator_factory((axis,))

    # get types for field and output
    field_dtype = np.dtype(field_dtype).type
    if output_dtype is None:
        output_dtype = field_dtype
    else:
        output_dtype = np.dtype(output_dtype).type

    # select kernel function implementation based on accumulate flag
    kernel_function = _lagrange_1D_kernel_function_factory(N, D, accumulate)

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
    N: tuple[int, int] = (1, 1),
    D: tuple[int, int] = (0, 0),
    accumulate: bool = False,
) -> BoundKernel:
    """Construct a 2D interpolation kernel for the specified axes.

    Parameters
    ----------
    axes : tuple[ArrayAxis or str, ArrayAxis or str]
        Tuple of two axes to perform interpolation along. If strings are provided, they must be one of "Z", "Y", or "X".
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    field_dtype : npt.DTypeLike, optional
        Data type of the input field, by default np.float64.
    output_dtype : npt.DTypeLike | None, optional
        Data type of the output particle property, if `None` (default) equal to the `field_dtype`.
    N : tuple[int, int], optional
        Stencil half width in the two dimensions for the interpolation polynomial, by default (1, 1) corresponding to bilinear interpolation.
    D : tuple[int, int], optional
        Derivative order of the Lagrange interpolating polynomial in the two dimensions. Default is (0, 0), which corresponds to value interpolation.
        Higher values correspond to derivatives with respect to index space.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.

    Returns
    -------
    BoundKernel
        A BoundKernel for performing 2D interpolation along the specified axes.

    Raises
    ------
    ValueError
        If the `axes`, `N` or `D` tuples do not contain exactly two elements.
        If the two axes are not different.
        If either axis is not a valid :class:`ArrayAxis`.
        If either element of `N` is not a positive integer.
        If either element of `D` is not a non-negative integer.
    """
    if len(axes) != 2:
        raise ValueError("axes must be a tuple of two elements.")
    if len(N) != 2:
        raise ValueError("N must be a tuple of two integers.")
    if len(D) != 2:
        raise ValueError("D must be a tuple of two integers.")

    axis_0 = ArrayAxis.parse(axes[0])
    axis_1 = ArrayAxis.parse(axes[1])

    if axis_0 == axis_1:
        raise ValueError(f"The two axes must be different. Received: {axis_0} and {axis_1}.")

    # get index names and validators for the specified axes
    idx_name_0 = axis_0.particle_index_name
    idx_name_1 = axis_1.particle_index_name
    validator = ordering_validator_factory((axis_0, axis_1))

    # get types for field and output
    field_dtype = np.dtype(field_dtype).type
    if output_dtype is None:
        output_dtype = field_dtype
    else:
        output_dtype = np.dtype(output_dtype).type

    # select kernel function implementation
    kernel_function = _lagrange_2D_kernel_function_factory(*N, *D, accumulate)

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
    N: tuple[int, int, int] = (1, 1, 1),
    D: tuple[int, int, int] = (0, 0, 0),
    accumulate: bool = False,
) -> BoundKernel:
    """Construct a 3D interpolation kernel.

    Parameters
    ----------
    axes : tuple[ArrayAxis or str, ArrayAxis or str, ArrayAxis or str]
        Tuple of three axes to perform interpolation along. If strings are provided, they must be one of "Z", "Y", or "X".
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    field_dtype : npt.DTypeLike, optional
        Data type of the input field, by default np.float64.
    output_dtype : npt.DTypeLike | None, optional
        Data type of the output particle property, if `None` (default) equal to `field_dtype`.
    N : tuple[int, int, int], optional
        Stencil half width in the three dimensions for the interpolation polynomial, by default (1, 1, 1) corresponding to trilinear interpolation.
    D : tuple[int, int, int], optional
        Derivative order of the Lagrange interpolating polynomial in the three dimensions. Default is (0, 0, 0), which corresponds to value interpolation.
        Higher values correspond to derivatives with respect to index space.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.

    Returns
    -------
    BoundKernel
        A BoundKernel for performing 3D interpolation along the specified axes.

    Raises
    ------
    ValueError
        If the `axes`, `N` or `D` tuples do not contain exactly three elements.
        If any of the axes are the same.
        If any axis is not a valid :class:`ArrayAxis`.
        If any element of `N` is not a positive integer.
        If any element of `D` is not a non-negative integer.
    """
    if len(axes) != 3:
        raise ValueError("axes must be a tuple of three elements.")
    if len(N) != 3:
        raise ValueError("N must be a tuple of three integers.")
    if len(D) != 3:
        raise ValueError("D must be a tuple of three integers.")

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
    field_dtype = np.dtype(field_dtype).type
    if output_dtype is None:
        output_dtype = field_dtype
    else:
        output_dtype = np.dtype(output_dtype).type

    # select kernel function implementation
    kernel_function = _lagrange_3D_kernel_function_factory(*N, *D, accumulate)

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
construct_Z_interpolation_kernel = functools.partial(construct_1D_interpolation_kernel, ArrayAxis.Z)
#: Partial function application of :func:`construct_1D_interpolation_kernel` with axis=ArrayAxis.Y.
construct_Y_interpolation_kernel = functools.partial(construct_1D_interpolation_kernel, ArrayAxis.Y)
#: Partial function application of :func:`construct_1D_interpolation_kernel` with axis=ArrayAxis.X.
construct_X_interpolation_kernel = functools.partial(construct_1D_interpolation_kernel, ArrayAxis.X)
#: Partial function application of :func:`construct_2D_interpolation_kernel` with axes=(ArrayAxis.X, ArrayAxis.Y).
construct_XY_interpolation_kernel = functools.partial(construct_2D_interpolation_kernel, (ArrayAxis.X, ArrayAxis.Y))
#: Partial function application of :func:`construct_2D_interpolation_kernel` with axes=(ArrayAxis.X, ArrayAxis.Z).
construct_XZ_interpolation_kernel = functools.partial(construct_2D_interpolation_kernel, (ArrayAxis.X, ArrayAxis.Z))
#: Partial function application of :func:`construct_2D_interpolation_kernel` with axes=(ArrayAxis.Y, ArrayAxis.X).
construct_YX_interpolation_kernel = functools.partial(construct_2D_interpolation_kernel, (ArrayAxis.Y, ArrayAxis.X))
#: Partial function application of :func:`construct_2D_interpolation_kernel` with axes=(ArrayAxis.Y, ArrayAxis.Z).
construct_YZ_interpolation_kernel = functools.partial(construct_2D_interpolation_kernel, (ArrayAxis.Y, ArrayAxis.Z))
#: Partial function application of :func:`construct_2D_interpolation_kernel` with axes=(ArrayAxis.Z, ArrayAxis.X).
construct_ZX_interpolation_kernel = functools.partial(construct_2D_interpolation_kernel, (ArrayAxis.Z, ArrayAxis.X))
#: Partial function application of :func:`construct_2D_interpolation_kernel` with axes=(ArrayAxis.Z, ArrayAxis.Y).
construct_ZY_interpolation_kernel = functools.partial(construct_2D_interpolation_kernel, (ArrayAxis.Z, ArrayAxis.Y))

#: Partial function application of :func:`construct_3D_interpolation_kernel` with axes=(ArrayAxis.X, ArrayAxis.Y, ArrayAxis.Z).
construct_XYZ_interpolation_kernel = functools.partial(
    construct_3D_interpolation_kernel, (ArrayAxis.X, ArrayAxis.Y, ArrayAxis.Z)
)
#: Partial function application of :func:`construct_3D_interpolation_kernel` with axes=(ArrayAxis.X, ArrayAxis.Z, ArrayAxis.Y).
construct_XZY_interpolation_kernel = functools.partial(
    construct_3D_interpolation_kernel, (ArrayAxis.X, ArrayAxis.Z, ArrayAxis.Y)
)
#: Partial function application of :func:`construct_3D_interpolation_kernel` with axes=(ArrayAxis.Y, ArrayAxis.X, ArrayAxis.Z).
construct_YXZ_interpolation_kernel = functools.partial(
    construct_3D_interpolation_kernel, (ArrayAxis.Y, ArrayAxis.X, ArrayAxis.Z)
)
#: Partial function application of :func:`construct_3D_interpolation_kernel` with axes=(ArrayAxis.Y, ArrayAxis.Z, ArrayAxis.X).
construct_YZX_interpolation_kernel = functools.partial(
    construct_3D_interpolation_kernel, (ArrayAxis.Y, ArrayAxis.Z, ArrayAxis.X)
)
#: Partial function application of :func:`construct_3D_interpolation_kernel` with axes=(ArrayAxis.Z, ArrayAxis.X, ArrayAxis.Y).
construct_ZXY_interpolation_kernel = functools.partial(
    construct_3D_interpolation_kernel, (ArrayAxis.Z, ArrayAxis.X, ArrayAxis.Y)
)
#: Partial function application of :func:`construct_3D_interpolation_kernel` with axes=(ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X).
construct_ZYX_interpolation_kernel = functools.partial(
    construct_3D_interpolation_kernel, (ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X)
)
