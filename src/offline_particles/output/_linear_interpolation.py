"""Generate output by linearly interpolating field data to particle positions."""

from ..fieldset import Fieldset
from ..kernels.output import (
    bilinear_interpolation_kernel,
    linear_interpolation_kernel,
    trilinear_interpolation_kernel,
)
from ._output import Output


def linearly_interpolate_fields(
    fieldset: Fieldset,
    *variables: str,
    temporary_array_prefix: str = "_output",
) -> list[Output]:
    """Output variables that linearly interpolate field data.

    Args:
        fieldset: The fieldset containing the fields to interpolate.
        variables: The list of variable names to interpolate.
        temporary_array_prefix: The prefix for temporary arrays to store the output data on particles.
    """
    dims = ("z", "y", "x")
    outputs = []

    for var in variables:
        if var not in fieldset:
            raise KeyError(f"Field '{var}' not found in fieldset.")

        field = fieldset[var]
        dmask = field.dmask
        ndim = field.nspatial_dims
        dtype = field.dtype
        tmp_name = f"{temporary_array_prefix}_{dtype}"

        if ndim == 1:
            dim = dims[dmask.index(True)]
            kernel = linear_interpolation_kernel(var, tmp_name, dim)
        elif ndim == 2:
            dim = tuple(dims[i] for i in range(3) if dmask[i])
            kernel = bilinear_interpolation_kernel(var, tmp_name, dim)
        elif ndim == 3:
            kernel = trilinear_interpolation_kernel(var, tmp_name)
        else:
            raise ValueError(
                f"Field '{var}' has unsupported number of dimensions: {ndim}"
            )

        outputs.append(Output(var, tmp_name, kernel))

    return outputs
