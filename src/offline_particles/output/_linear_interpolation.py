"""Generate output by linearly interpolating field data to particle positions."""

from ..fieldset import Fieldset
from ..kernels.output import (
    bilinear_interpolation_kernel,
    linear_interpolation_kernel,
    trilinear_interpolation_kernel,
)
from ._output import Output

DMASK_DIM_MAPPING_2D = {
    (True, True, False): ("z", "y"),
    (True, False, True): ("z", "x"),
    (False, True, True): ("y", "x"),
}


def linearly_interpolate_fields(
    fieldset: Fieldset,
    *variables: str,
    particle_field_prefix: str = "_output",
) -> list[Output]:
    """Output variables that linearly interpolate field data.

    Args:
        fieldset: The fieldset containing the fields to interpolate.
        variables: The list of variable names to interpolate.
        particle_field_prefix: The prefix for the particle array to store the output data.
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
        particle_field = f"{particle_field_prefix}_{dtype}"

        if ndim == 1:
            dim = dims[dmask.index(True)]
            kernel = linear_interpolation_kernel(var, particle_field, dim)
        elif ndim == 2:
            dim = DMASK_DIM_MAPPING_2D[dmask]
            kernel = bilinear_interpolation_kernel(var, particle_field, dim)
        elif ndim == 3:
            kernel = trilinear_interpolation_kernel(var, particle_field)
        else:
            raise ValueError(f"Field '{var}' has unsupported number of dimensions: {ndim}")

        outputs.append(Output(var, kernel, particle_field=particle_field))

    return outputs
