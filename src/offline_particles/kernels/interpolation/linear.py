"""Linear interpolation kernels."""

import functools

import numba
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
from ..status import INACTIVE_FLAG

__all__ = [
    "construct_linear_interpolation_kernel",
    "construct_bilinear_interpolation_kernel",
    "construct_trilinear_interpolation_kernel",
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


@numba.njit(nogil=True, fastmath=True)
def _truncate_index(idx: float, max_idx: int) -> int:
    """Truncate the index to be within the bounds of the field array."""
    idx = int(idx)  # floor the index to get the lower index
    if idx < 0:
        return 0
    elif idx > max_idx:
        return max_idx
    else:
        return idx


########################
# linear interpolation #
########################


@numba.njit(parallel=True, nogil=True, fastmath=True)
def _linear_interpolation(
    status: npt.NDArray[np.uint8],
    idx: npt.NDArray[np.float64],
    output: npt.NDArray[np.generic],
    field: npt.NDArray[np.generic],
    offset: np.float64,
    accumulate: bool = False,
) -> None:
    """Implementation of a 1D linear interpolation kernel."""
    max_idx = field.shape[0] - 2  # max index for the lower index to avoid out-of-bounds
    for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
        if status[i] & INACTIVE_FLAG:
            continue
        offset_idx = idx[i] + offset
        I0 = _truncate_index(offset_idx, max_idx)
        f0 = offset_idx - I0
        g0 = 1.0 - f0
        value = g0 * field[I0] + f0 * field[I0 + 1]
        if accumulate:
            output[i] += value
        else:
            output[i] = value


def linear_interpolation_kernel_function(
    particle_properties: ParticlePropertiesType,
    scalars: ScalarsType,
    fields: FieldDataType,
) -> None:
    field_data = fields["field"]
    _linear_interpolation(
        particle_properties["status"],
        particle_properties["idx"],
        particle_properties["output"],
        field_data.array,
        field_data.offsets[0],
        accumulate=False,
    )


def linear_interpolation_accumulation_kernel_function(
    particle_properties: ParticlePropertiesType,
    scalars: ScalarsType,
    fields: FieldDataType,
) -> None:
    field_data = fields["field"]
    _linear_interpolation(
        particle_properties["status"],
        particle_properties["idx"],
        particle_properties["output"],
        field_data.array,
        field_data.offsets[0],
        accumulate=True,
    )


def construct_linear_interpolation_kernel(
    axis: ArrayAxis | str,
    output: str,
    field: str,
    field_dtype: npt.DTypeLike = np.float64,
    output_dtype: npt.DTypeLike | None = None,
    accumulate: bool = False,
) -> BoundKernel:
    """Construct a linear interpolation kernel for the specified axis.

    Parameters
    ----------
    axis : ArrayAxis or str
        Axis to perform linear interpolation along. If a string is provided, it must be one of "Z", "Y", or "X" (case-insensitive).
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    field_dtype : npt.DTypeLike, optional
        Data type of the input field, by default np.float64.
    output_dtype : npt.DTypeLike | None, optional
        Data type of the output particle property, if None (default), field_dtype is used.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.

    Returns
    -------
    BoundKernel
        A BoundKernel for performing linear interpolation along the specified axis.
    """
    axis = ArrayAxis.parse(axis)  # parse the axis argument to an ArrayAxis enum member
    idx_name = axis.particle_index_name
    validator = ordering_validator_factory((axis,))

    if accumulate:
        kernel_function = linear_interpolation_accumulation_kernel_function
    else:
        kernel_function = linear_interpolation_kernel_function

    particle_kernel = ParticleKernel(
        kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("idx", np.float64),
            ParticlePropertyDeclaration("output", np.dtype(output_dtype)),
        ],
        field_data=[
            FieldDataDeclaration("field", np.float64, [validator]),
        ],
    )

    return BoundKernel(
        particle_kernel,
        particle_property_bindings={
            "idx": idx_name,
            "output": output,
        },
        field_data_bindings={
            "field": field,
        },
    )


#########################
# blinear interpolation #
#########################


@numba.njit(parallel=True, nogil=True, fastmath=True)
def _bilinear_interpolation(
    status: npt.NDArray[np.uint8],
    idx_0: npt.NDArray[np.float64],
    idx_1: npt.NDArray[np.float64],
    output: npt.NDArray[np.generic],
    field: npt.NDArray[np.generic],
    offset_0: np.float64,
    offset_1: np.float64,
    accumulate: bool = False,
) -> None:
    """Implementation of a 2D bilinear interpolation kernel."""
    max_idx_0 = field.shape[0] - 2
    max_idx_1 = field.shape[1] - 2
    for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
        if status[i] & INACTIVE_FLAG:
            continue
        offset_idx_0 = idx_0[i] + offset_0
        offset_idx_1 = idx_1[i] + offset_1
        I0 = _truncate_index(offset_idx_0, max_idx_0)
        I1 = _truncate_index(offset_idx_1, max_idx_1)

        f0 = offset_idx_0 - I0
        f1 = offset_idx_1 - I1
        g0 = 1.0 - f0
        g1 = 1.0 - f1

        v00 = field[I0, I1]
        v01 = field[I0, I1 + 1]
        v10 = field[I0 + 1, I1]
        v11 = field[I0 + 1, I1 + 1]

        value = g0 * g1 * v00 + g0 * f1 * v01 + f0 * g1 * v10 + f0 * f1 * v11

        if accumulate:
            output[i] += value
        else:
            output[i] = value


def bilinear_interpolation_kernel_function(
    particle_properties: ParticlePropertiesType,
    scalars: ScalarsType,
    fields: FieldDataType,
) -> None:
    field_data = fields["field"]
    _bilinear_interpolation(
        particle_properties["status"],
        particle_properties["idx_0"],
        particle_properties["idx_1"],
        particle_properties["output"],
        field_data.array,
        field_data.offsets[0],
        field_data.offsets[1],
        accumulate=False,
    )


def bilinear_interpolation_accumulation_kernel_function(
    particle_properties: ParticlePropertiesType,
    scalars: ScalarsType,
    fields: FieldDataType,
) -> None:
    field_data = fields["field"]
    _bilinear_interpolation(
        particle_properties["status"],
        particle_properties["idx_0"],
        particle_properties["idx_1"],
        particle_properties["output"],
        field_data.array,
        field_data.offsets[0],
        field_data.offsets[1],
        accumulate=True,
    )


def construct_bilinear_interpolation_kernel(
    axes: tuple[ArrayAxis | str, ArrayAxis | str],
    output: str,
    field: str,
    field_dtype: npt.DTypeLike = np.float64,
    output_dtype: npt.DTypeLike | None = None,
    accumulate: bool = False,
) -> BoundKernel:
    """Construct a bilinear interpolation kernel for the specified axes.

    Parameters
    ----------
    axes : tuple[ArrayAxis or str, ArrayAxis or str]
        Tuple of two axes to perform bilinear interpolation along. If strings are provided, they must be one of "Z", "Y", or "X" (case-insensitive).
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    field_dtype : npt.DTypeLike, optional
        Data type of the input field, by default np.float64.
    output_dtype : npt.DTypeLike | None, optional
        Data type of the output particle property, if None (default) equal to the field_dtype.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.

    Returns
    -------
    BoundKernel
        A BoundKernel for performing bilinear interpolation along the specified axes.
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

    # select kernel function based on accumulate flag
    if accumulate:
        kernel_function = bilinear_interpolation_accumulation_kernel_function
    else:
        kernel_function = bilinear_interpolation_kernel_function

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

    return BoundKernel(
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


###########################
# trilinear interpolation #
###########################


@numba.njit(parallel=True, nogil=True, fastmath=True)
def _trilinear_interpolation(
    status: npt.NDArray[np.uint8],
    idx_0: npt.NDArray[np.float64],
    idx_1: npt.NDArray[np.float64],
    idx_2: npt.NDArray[np.float64],
    output: npt.NDArray[np.generic],
    field: npt.NDArray[np.generic],
    offset_0: np.float64,
    offset_1: np.float64,
    offset_2: np.float64,
    accumulate: bool = False,
) -> None:
    """Implementation of a 3D trilinear interpolation kernel."""
    max_idx_0 = field.shape[0] - 2
    max_idx_1 = field.shape[1] - 2
    max_idx_2 = field.shape[2] - 2
    for i in numba.prange(status.shape[0]):  # type: ignore[not-iterable]
        if status[i] & INACTIVE_FLAG:
            continue
        offset_idx_0 = idx_0[i] + offset_0
        offset_idx_1 = idx_1[i] + offset_1
        offset_idx_2 = idx_2[i] + offset_2
        I0 = _truncate_index(offset_idx_0, max_idx_0)
        I1 = _truncate_index(offset_idx_1, max_idx_1)
        I2 = _truncate_index(offset_idx_2, max_idx_2)

        f0 = offset_idx_0 - I0
        f1 = offset_idx_1 - I1
        f2 = offset_idx_2 - I2
        g0 = 1.0 - f0
        g1 = 1.0 - f1
        g2 = 1.0 - f2

        v000 = field[I0, I1, I2]
        v001 = field[I0, I1, I2 + 1]
        v010 = field[I0, I1 + 1, I2]
        v011 = field[I0, I1 + 1, I2 + 1]
        v100 = field[I0 + 1, I1, I2]
        v101 = field[I0 + 1, I1, I2 + 1]
        v110 = field[I0 + 1, I1 + 1, I2]
        v111 = field[I0 + 1, I1 + 1, I2 + 1]

        value = (
            g0 * g1 * g2 * v000
            + g0 * g1 * f2 * v001
            + g0 * f1 * g2 * v010
            + g0 * f1 * f2 * v011
            + f0 * g1 * g2 * v100
            + f0 * g1 * f2 * v101
            + f0 * f1 * g2 * v110
            + f0 * f1 * f2 * v111
        )

        if accumulate:
            output[i] += value
        else:
            output[i] = value


def trilinear_interpolation_kernel_function(
    particle_properties: ParticlePropertiesType,
    scalars: ScalarsType,
    fields: FieldDataType,
) -> None:
    field_data = fields["field"]
    _trilinear_interpolation(
        particle_properties["status"],
        particle_properties["idx_0"],
        particle_properties["idx_1"],
        particle_properties["idx_2"],
        particle_properties["output"],
        field_data.array,
        field_data.offsets[0],
        field_data.offsets[1],
        field_data.offsets[2],
        accumulate=False,
    )


def trilinear_interpolation_accumulation_kernel_function(
    particle_properties: ParticlePropertiesType,
    scalars: ScalarsType,
    fields: FieldDataType,
) -> None:
    field_data = fields["field"]
    _trilinear_interpolation(
        particle_properties["status"],
        particle_properties["idx_0"],
        particle_properties["idx_1"],
        particle_properties["idx_2"],
        particle_properties["output"],
        field_data.array,
        field_data.offsets[0],
        field_data.offsets[1],
        field_data.offsets[2],
        accumulate=True,
    )


def construct_trilinear_interpolation_kernel(
    axes: tuple[ArrayAxis | str, ArrayAxis | str, ArrayAxis | str],
    output: str,
    field: str,
    field_dtype: npt.DTypeLike = np.float64,
    output_dtype: npt.DTypeLike | None = None,
    accumulate: bool = False,
) -> BoundKernel:
    """Construct a trilinear interpolation kernel.

    Parameters
    ----------
    axes : tuple[ArrayAxis or str, ArrayAxis or str, ArrayAxis or str]
        Tuple of three axes to perform trilinear interpolation along. If strings are provided, they must be one of "Z", "Y", or "X" (case-insensitive).
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    field_dtype : npt.DTypeLike, optional
        Data type of the input field, by default np.float64.
    output_dtype : npt.DTypeLike | None, optional
        Data type of the output particle property, if None (default) equal to the field dtype.
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

    # select kernel function based on accumulate flag
    if accumulate:
        kernel_function = trilinear_interpolation_accumulation_kernel_function
    else:
        kernel_function = trilinear_interpolation_kernel_function

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

    return BoundKernel(
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


############################
# convenience constructors #
############################

#: Partial function application of :func:`construct_linear_interpolation_kernel` with axis=ArrayAxis.Z.
construct_Z_interpolation_kernel = functools.partial(construct_linear_interpolation_kernel, axis=ArrayAxis.Z)
#: Partial function application of :func:`construct_linear_interpolation_kernel` with axis=ArrayAxis.Y.
construct_Y_interpolation_kernel = functools.partial(construct_linear_interpolation_kernel, axis=ArrayAxis.Y)
#: Partial function application of :func:`construct_linear_interpolation_kernel` with axis=ArrayAxis.X.
construct_X_interpolation_kernel = functools.partial(construct_linear_interpolation_kernel, axis=ArrayAxis.X)

#: Partial function application of :func:`construct_bilinear_interpolation_kernel` with axes=(ArrayAxis.X, ArrayAxis.Y).
construct_XY_interpolation_kernel = functools.partial(
    construct_bilinear_interpolation_kernel, axes=(ArrayAxis.X, ArrayAxis.Y)
)
#: Partial function application of :func:`construct_bilinear_interpolation_kernel` with axes=(ArrayAxis.X, ArrayAxis.Z).
construct_XZ_interpolation_kernel = functools.partial(
    construct_bilinear_interpolation_kernel, axes=(ArrayAxis.X, ArrayAxis.Z)
)
#: Partial function application of :func:`construct_bilinear_interpolation_kernel` with axes=(ArrayAxis.Y, ArrayAxis.X).
construct_YX_interpolation_kernel = functools.partial(
    construct_bilinear_interpolation_kernel, axes=(ArrayAxis.Y, ArrayAxis.X)
)
#: Partial function application of :func:`construct_bilinear_interpolation_kernel` with axes=(ArrayAxis.Y, ArrayAxis.Z).
construct_YZ_interpolation_kernel = functools.partial(
    construct_bilinear_interpolation_kernel, axes=(ArrayAxis.Y, ArrayAxis.Z)
)
#: Partial function application of :func:`construct_bilinear_interpolation_kernel` with axes=(ArrayAxis.Z, ArrayAxis.X).
construct_ZX_interpolation_kernel = functools.partial(
    construct_bilinear_interpolation_kernel, axes=(ArrayAxis.Z, ArrayAxis.X)
)
#: Partial function application of :func:`construct_bilinear_interpolation_kernel` with axes=(ArrayAxis.Z, ArrayAxis.Y).
construct_ZY_interpolation_kernel = functools.partial(
    construct_bilinear_interpolation_kernel, axes=(ArrayAxis.Z, ArrayAxis.Y)
)

#: Partial function application of :func:`construct_trilinear_interpolation_kernel` with axes=(ArrayAxis.X, ArrayAxis.Y, ArrayAxis.Z).
construct_XYZ_interpolation_kernel = functools.partial(
    construct_trilinear_interpolation_kernel, axes=(ArrayAxis.X, ArrayAxis.Y, ArrayAxis.Z)
)
#: Partial function application of :func:`construct_trilinear_interpolation_kernel` with axes=(ArrayAxis.X, ArrayAxis.Z, ArrayAxis.Y).
construct_XZY_interpolation_kernel = functools.partial(
    construct_trilinear_interpolation_kernel, axes=(ArrayAxis.X, ArrayAxis.Z, ArrayAxis.Y)
)
#: Partial function application of :func:`construct_trilinear_interpolation_kernel` with axes=(ArrayAxis.Y, ArrayAxis.X, ArrayAxis.Z).
construct_YXZ_interpolation_kernel = functools.partial(
    construct_trilinear_interpolation_kernel, axes=(ArrayAxis.Y, ArrayAxis.X, ArrayAxis.Z)
)
#: Partial function application of :func:`construct_trilinear_interpolation_kernel` with axes=(ArrayAxis.Y, ArrayAxis.Z, ArrayAxis.X).
construct_YZX_interpolation_kernel = functools.partial(
    construct_trilinear_interpolation_kernel, axes=(ArrayAxis.Y, ArrayAxis.Z, ArrayAxis.X)
)
#: Partial function application of :func:`construct_trilinear_interpolation_kernel` with axes=(ArrayAxis.Z, ArrayAxis.X, ArrayAxis.Y).
construct_ZXY_interpolation_kernel = functools.partial(
    construct_trilinear_interpolation_kernel, axes=(ArrayAxis.Z, ArrayAxis.X, ArrayAxis.Y)
)
#: Partial function application of :func:`construct_trilinear_interpolation_kernel` with axes=(ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X).
construct_ZYX_interpolation_kernel = functools.partial(
    construct_trilinear_interpolation_kernel, axes=(ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X)
)
