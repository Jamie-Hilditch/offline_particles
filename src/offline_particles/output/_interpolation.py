"""Generate output by interpolating field data to particle positions using Lagrange polynomials."""

from collections.abc import Iterable

from ..fieldset import Fieldset
from ..kernels.interpolation import (
    construct_1D_interpolation_kernel,
    construct_2D_interpolation_kernel,
    construct_3D_interpolation_kernel,
)
from ._output import Output


def interpolate_fields(
    fieldset: Fieldset,
    variables: str | Iterable[str],
    N: int = 1,
    particle_property_prefix: str = "_output",
) -> dict[str, Output]:
    """Output variables that interpolate field data using Lagrange polynomials.

    Parameters
    ----------
    fieldset : Fieldset
        The fieldset containing the fields to interpolate.
    variables : str | Iterable[str]
        A string or an iterable of variable names to interpolate.
    N : int, optional
        The half width of the Lagrange interpolation stencil. Defaults to 1, which corresponds to linear interpolation.
    particle_property_prefix : str, optional
        The prefix for the particle array to store the output data, by default '_output'.

    Returns
    -------
    dict[str, Output]
        A dictionary mapping variable names to Output objects.

    Raises
    ------
    KeyError
        If a variable is not found in the fieldset.
    ValueError
        If a field has an unsupported number of axes.

    Notes
    -----
    Outputs are computed in the particle property named '{particle_property_prefix}_{field_dtype}', where 'field_dtype' is the data type
    of the field being interpolated. This particle property is assumed to be temporary and may be overwritten if multiple fields of the
    same data type are interpolated.
    """
    outputs = {}

    if isinstance(variables, str):
        variables = [variables]

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
                    N=N,
                )
            case (axis0, axis1):
                kernel = construct_2D_interpolation_kernel(
                    axes=(axis0, axis1),
                    output=particle_property,
                    field=var,
                    field_dtype=dtype,
                    output_dtype=dtype,
                    N=N,
                )
            case (axis0, axis1, axis2):
                kernel = construct_3D_interpolation_kernel(
                    axes=(axis0, axis1, axis2),
                    output=particle_property,
                    field=var,
                    field_dtype=dtype,
                    output_dtype=dtype,
                    N=N,
                )
            case _:
                raise ValueError(f"Field '{var}' has unsupported number of axes: {len(field.axes)}")

        outputs[var] = Output(particle_property, dtype=dtype, kernels=(kernel,))

    return outputs
