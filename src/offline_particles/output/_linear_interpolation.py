"""Generate output by linearly interpolating field data to particle positions."""

from ..fieldset import Fieldset
from ..kernels.interpolation import (
    construct_bilinear_interpolation_kernel,
    construct_linear_interpolation_kernel,
    construct_trilinear_interpolation_kernel,
)
from ..spatial_arrays import ArrayAxis
from ._output import Output


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
    outputs = {}

    for var in variables:
        if var not in fieldset:
            raise KeyError(f"Field '{var}' not found in fieldset.")

        field = fieldset[var]
        dtype = field.output_dtype
        particle_property = f"{particle_property_prefix}_{dtype}"

        match field.axes:
            case (ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X):
                kernel = construct_trilinear_interpolation_kernel(particle_property, var)
            case (ArrayAxis.Z, ArrayAxis.Y):
                kernel = construct_bilinear_interpolation_kernel(("zidx", "yidx"), particle_property, var)
            case (ArrayAxis.Z, ArrayAxis.X):
                kernel = construct_bilinear_interpolation_kernel(("zidx", "xidx"), particle_property, var)
            case (ArrayAxis.Y, ArrayAxis.X):
                kernel = construct_bilinear_interpolation_kernel(("yidx", "xidx"), particle_property, var)
            case (ArrayAxis.Z,):
                kernel = construct_linear_interpolation_kernel("zidx", particle_property, var)
            case (ArrayAxis.Y,):
                kernel = construct_linear_interpolation_kernel("yidx", particle_property, var)
            case (ArrayAxis.X,):
                kernel = construct_linear_interpolation_kernel("xidx", particle_property, var)
            case _:
                raise ValueError(f"Field '{var}' has unsupported axes: {field.axes}")

        name = f"{particle_set}:{var}"
        outputs[name] = Output(particle_set, particle_property, kernel)

    return outputs
