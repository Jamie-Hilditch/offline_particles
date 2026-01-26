"""Linear interpolation kernels."""

from typing import Literal

import numpy as np

from ...spatial_arrays import ACTIVE_STAGGERS, INACTIVE_STAGGERS
from .._kernels import FieldDataDeclaration, KernelBinding, ParticleKernel, ParticlePropertyDeclaration
from ..common_inputs import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from ._linear import (
    bilinear_interpolation_accumulation_kernel_function,
    bilinear_interpolation_kernel_function,
    linear_interpolation_accumulation_kernel_function,
    linear_interpolation_kernel_function,
    trilinear_interpolation_accumulation_kernel_function,
    trilinear_interpolation_kernel_function,
)

__all__ = [
    "construct_linear_interpolation_kernel",
    "construct_bilinear_interpolation_kernel",
    "construct_trilinear_interpolation_kernel",
    "construct_vertical_interpolation_kernel",
    "construct_horizontal_interpolation_kernel",
]

output_declaration = ParticlePropertyDeclaration("output", np.float64)
field_data_declarations_1d = {
    "zidx": FieldDataDeclaration(
        "field", np.float64, z_staggers=ACTIVE_STAGGERS, y_staggers=INACTIVE_STAGGERS, x_staggers=INACTIVE_STAGGERS
    ),
    "yidx": FieldDataDeclaration(
        "field", np.float64, z_staggers=INACTIVE_STAGGERS, y_staggers=ACTIVE_STAGGERS, x_staggers=INACTIVE_STAGGERS
    ),
    "xidx": FieldDataDeclaration(
        "field", np.float64, z_staggers=INACTIVE_STAGGERS, y_staggers=INACTIVE_STAGGERS, x_staggers=ACTIVE_STAGGERS
    ),
}
field_data_declarations_2d = {
    ("zidx", "yidx"): FieldDataDeclaration(
        "field", np.float64, z_staggers=ACTIVE_STAGGERS, y_staggers=ACTIVE_STAGGERS, x_staggers=INACTIVE_STAGGERS
    ),
    ("zidx", "xidx"): FieldDataDeclaration(
        "field", np.float64, z_staggers=ACTIVE_STAGGERS, y_staggers=INACTIVE_STAGGERS, x_staggers=ACTIVE_STAGGERS
    ),
    ("yidx", "xidx"): FieldDataDeclaration(
        "field", np.float64, z_staggers=INACTIVE_STAGGERS, y_staggers=ACTIVE_STAGGERS, x_staggers=ACTIVE_STAGGERS
    ),
}


def _linear_interpolation_kernel(field_data_declaration: FieldDataDeclaration) -> ParticleKernel:
    return ParticleKernel(
        linear_interpolation_kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("idx", np.float64),
            output_declaration,
        ],
        field_data=[
            field_data_declaration,
        ],
    )


def _bilinear_interpolation_kernel(field_data_declaration: FieldDataDeclaration) -> ParticleKernel:
    return ParticleKernel(
        bilinear_interpolation_kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("idx_0", np.float64),
            ParticlePropertyDeclaration("idx_1", np.float64),
            output_declaration,
        ],
        field_data=[
            field_data_declaration,
        ],
    )


def _trilinear_interpolation_kernel() -> ParticleKernel:
    return ParticleKernel(
        trilinear_interpolation_kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ZIDX_DECLARATION,
            YIDX_DECLARATION,
            XIDX_DECLARATION,
            output_declaration,
        ],
        field_data=[
            FieldDataDeclaration(
                "field", np.float64, zstaggers=ACTIVE_STAGGERS, y_staggers=ACTIVE_STAGGERS, x_staggers=ACTIVE_STAGGERS
            ),
        ],
    )


def _linear_interpolation_accumulation_kernel(field_data_declaration: FieldDataDeclaration) -> ParticleKernel:
    return ParticleKernel(
        linear_interpolation_accumulation_kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("idx", np.float64),
            output_declaration,
        ],
        field_data=[
            field_data_declaration,
        ],
    )


def _bilinear_interpolation_accumulation_kernel(field_data_declaration: FieldDataDeclaration) -> ParticleKernel:
    return ParticleKernel(
        bilinear_interpolation_accumulation_kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ParticlePropertyDeclaration("idx_0", np.float64),
            ParticlePropertyDeclaration("idx_1", np.float64),
            output_declaration,
        ],
        field_data=[
            field_data_declaration,
        ],
    )


def _trilinear_interpolation_accumulation_kernel() -> ParticleKernel:
    return ParticleKernel(
        trilinear_interpolation_accumulation_kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ZIDX_DECLARATION,
            YIDX_DECLARATION,
            XIDX_DECLARATION,
            output_declaration,
        ],
        field_data=[
            FieldDataDeclaration(
                "field", np.float64, zstaggers=ACTIVE_STAGGERS, y_staggers=ACTIVE_STAGGERS, x_staggers=ACTIVE_STAGGERS
            ),
        ],
    )


# some KernelBinding factories
def construct_linear_interpolation_kernel(
    idx: Literal["zidx", "xidx", "yidx"], output: str, field: str, accumulate: bool = False
) -> KernelBinding:
    """Linear interpolation kernel.

    Parameters
    ----------
    idx : Literal["zidx", "xidx", "yidx"]
        Dimension index to bind the `idx` to.
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.
    """
    if idx not in ("zidx", "xidx", "yidx"):
        raise ValueError("idx must be one of 'zidx', 'xidx', or 'yidx'.")
    if accumulate:
        kernel = _linear_interpolation_accumulation_kernel(field_data_declarations_1d[idx])
    else:
        kernel = _linear_interpolation_kernel(field_data_declarations_1d[idx])
    return KernelBinding(
        kernel,
        particle_property_bindings={
            "idx": idx,
            "output": output,
        },
        field_data_bindings={
            "field": field,
        },
    )


def construct_vertical_interpolation_kernel(output: str, field: str, accumulate: bool = False) -> KernelBinding:
    """Vertical linear interpolation kernel.

    A linear interpolation kernel with `idx` bound to `zidx` and `output`
    and `field` bound to the provided names.

    Parameters
    ----------
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.
    """
    return construct_linear_interpolation_kernel("zidx", output, field, accumulate)


def construct_bilinear_interpolation_kernel(
    indices: tuple[str, str], output: str, field: str, accumulate: bool = False
) -> KernelBinding:
    """Bilinear interpolation kernel.

    Parameters
    ----------
    indices : tuple[str,str]
        Tuple of dimension indices to bind `idx_0` and `idx_1` to.
        Valid values: ("zidx", "yidx"), ("zidx", "xidx"), ("yidx", "xidx")
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.
    """
    if indices not in (("zidx", "yidx"), ("zidx", "xidx"), ("yidx", "xidx")):
        raise ValueError("indices must be one of ('zidx', 'yidx'), ('zidx', 'xidx'), or ('yidx', 'xidx').")
    if accumulate:
        kernel = _bilinear_interpolation_accumulation_kernel(field_data_declarations_2d[indices])
    else:
        kernel = _bilinear_interpolation_kernel(field_data_declarations_2d[indices])
    return KernelBinding(
        kernel,
        particle_property_bindings={
            "idx_0": indices[0],
            "idx_1": indices[1],
            "output": output,
        },
        field_data_bindings={
            "field": field,
        },
    )


def construct_horizontal_interpolation_kernel(output: str, field: str, accumulate: bool = False) -> KernelBinding:
    """Horizontal bilinear interpolation kernel.

    A bilinear interpolation kernel with `idx_0` and `idx_1` bound to `yidx` and `xidx` respectively,
    and `output` and `field` bound to the provided names.

    Parameters
    ----------
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.
    """
    return construct_bilinear_interpolation_kernel(("yidx", "xidx"), output, field, accumulate)


def construct_trilinear_interpolation_kernel(output: str, field: str, accumulate: bool = False) -> KernelBinding:
    """Trilinear interpolation kernel.

    Parameters
    ----------
    output : str
        Name of the particle property to bind the output to.
    field : str
        Name of the field data to bind the input field to.
    accumulate : bool, optional
        Whether the kernel accumulates to or overwrites the output property, by default False.
    """
    if accumulate:
        kernel = _trilinear_interpolation_accumulation_kernel()
    else:
        kernel = _trilinear_interpolation_kernel()
    return KernelBinding(
        kernel,
        particle_property_bindings={
            "output": output,
        },
        field_data_bindings={
            "field": field,
        },
    )
