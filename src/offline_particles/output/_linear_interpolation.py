"""Generate output by linearly interpolating field data to particle positions."""

from ..fieldset import Fieldset
from ..kernels.interpolation import (
    construct_1D_interpolation_kernel,
    construct_2D_interpolation_kernel,
    construct_3D_interpolation_kernel,
)
from ._output import Output


def linearly_interpolate_fields(
    fieldset: Fieldset,
    *variables: str,
    particle_property_prefix: str = "_output",
) -> dict[str, Output]:
    """Output variables that linearly interpolate field data.

    Args:
        fieldset: The fieldset containing the fields to interpolate.
        variables: The list of variable names to interpolate.
        particle_property_prefix: The prefix for the particle array to store the output data.
    """
    outputs = {}

    for var in variables:
        if var not in fieldset:
            raise KeyError(f"Field '{var}' not found in fieldset.")

        field = fieldset[var]
        dtype = field.output_dtype
        particle_property = f"{particle_property_prefix}_{dtype}"

        match field.axes:
            case (axis,):
                kernel = construct_1D_interpolation_kernel(
                    axis=axis,
                    output=particle_property,
                    field=var,
                    field_dtype=dtype,
                    output_dtype=dtype,
                )
            case (axis0, axis1):
                kernel = construct_2D_interpolation_kernel(
                    axes=(axis0, axis1),
                    output=particle_property,
                    field=var,
                    field_dtype=dtype,
                    output_dtype=dtype,
                )
            case (axis0, axis1, axis2):
                kernel = construct_3D_interpolation_kernel(
                    axes=(axis0, axis1, axis2),
                    output=particle_property,
                    field=var,
                    field_dtype=dtype,
                    output_dtype=dtype,
                )
            case _:
                raise ValueError(f"Field '{var}' has unsupported number of axes: {len(field.axes)}")

        outputs[var] = Output(particle_property, kernel)

    return outputs
