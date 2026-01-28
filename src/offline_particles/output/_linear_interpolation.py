"""Generate output by linearly interpolating field data to particle positions."""

from ..fieldset import Fieldset
from ..kernels.interpolation import (
    construct_bilinear_interpolation_kernel,
    construct_linear_interpolation_kernel,
    construct_trilinear_interpolation_kernel,
)
from ._output import Output

DMASK_DIM_MAPPING_2D = {
    (True, True, False): ("zidx", "yidx"),
    (True, False, True): ("zidx", "xidx"),
    (False, True, True): ("yidx", "xidx"),
}


def linearly_interpolate_fields(
    fieldset: Fieldset,
    particle_set: str,
    *variables: str,
    particle_property_prefix: str = "_output",
) -> dict[str, Output]:
    """Output variables that linearly interpolate field data.

    Args:
        fieldset: The fieldset containing the fields to interpolate.
        variables: The list of variable names to interpolate.
        particle_field_prefix: The prefix for the particle array to store the output data.
    """
    dims = ("zidx", "yidx", "xidx")
    outputs = {}

    for var in variables:
        if var not in fieldset:
            raise KeyError(f"Field '{var}' not found in fieldset.")

        field = fieldset[var]
        dmask = field.dmask
        ndim = field.nspatial_dims
        dtype = field.output_dtype
        particle_property = f"{particle_property_prefix}_{dtype}"

        if ndim == 1:
            dim = dims[dmask.index(True)]
            kernel = construct_linear_interpolation_kernel(dim, particle_property, var)
        elif ndim == 2:
            dim = DMASK_DIM_MAPPING_2D[dmask]
            kernel = construct_bilinear_interpolation_kernel(dim, particle_property, var)
        elif ndim == 3:
            kernel = construct_trilinear_interpolation_kernel(particle_property, var)
        else:
            raise ValueError(f"Field '{var}' has unsupported number of dimensions: {ndim}")

        name = f"{particle_set}:{var}"
        outputs[name] = Output(particle_set, particle_property, kernel)

    return outputs
