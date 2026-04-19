"""Submodule for saving simulation output."""

from ._linear_interpolation import linearly_interpolate_fields
from ._output import AbstractOutputWriter, AbstractOutputWriterBuilder, Output
from ._zarr_writer import ZarrOutputBuilder, ZarrOutputWriter

__all__ = [
    "AbstractOutputWriter",
    "AbstractOutputWriterBuilder",
    "Output",
    "ZarrOutputWriter",
    "ZarrOutputBuilder",
    "linearly_interpolate_fields",
]

# Set __module__ for all public classes to this module for cleaner documentation
_module = __name__
for _obj in [
    AbstractOutputWriter,
    AbstractOutputWriterBuilder,
    Output,
    ZarrOutputWriter,
    ZarrOutputBuilder,
    linearly_interpolate_fields,
]:
    _obj.__module__ = _module
