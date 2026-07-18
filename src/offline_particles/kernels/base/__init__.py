"""Some base kernels for doing simple operations on particles."""

import numpy as np

from .._kernels import BoundKernel, ParticleKernel, ParticlePropertyDeclaration
from ..input_declarations import STATUS_DECLARATION
from ._base import (
    add_property,
    copy_property,
    divide_property,
    multiply_property,
    subtract_property,
)

__all__ = [
    "construct_add_property_kernel",
    "construct_copy_property_kernel",
    "construct_divide_property_kernel",
    "construct_multiply_property_kernel",
    "construct_subtract_property_kernel",
]

type SupportedDTypes = type[np.float32 | np.float64 | np.integer]


def construct_copy_property_kernel(
    source: str,
    destination: str,
    dtype: SupportedDTypes = np.float64,
) -> BoundKernel:
    """Construct a kernel to copy a particle property from source to destination.

    Parameters
    ----------
    source : str
        Binding for the source particle property.
    destination : str
        Binding for the destination particle property.
    dtype : type[np.float32] | type[np.float64] | type[np.integer], optional
        Data type of the particle properties (default np.float64).
        Supported types are np.float32, np.float64, and any np.integer.

    Returns
    -------
    BoundKernel
        A bound kernel that copies the source property to the destination property.
    """
    source_declaration = ParticlePropertyDeclaration("source", np.dtype(dtype).type)
    destination_declaration = ParticlePropertyDeclaration("destination", np.dtype(dtype).type)

    kernel = ParticleKernel(
        copy_property,
        particle_properties=[
            STATUS_DECLARATION,
            source_declaration,
            destination_declaration,
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "source": source,
            "destination": destination,
        },
    )


def construct_add_property_kernel(
    source: str,
    destination: str,
    dtype: SupportedDTypes = np.float64,
) -> BoundKernel:
    """Construct a kernel to add particle property source to destination.

    Parameters
    ----------
    source : str
        Binding for the source particle property.
    destination : str
        Binding for the destination particle property.
    dtype : type[np.float32] | type[np.float64] | type[np.integer], optional
        Data type of the particle properties (default np.float64).
        Supported types are np.float32, np.float64, and any np.integer.

    Returns
    -------
    BoundKernel
        A bound kernel that adds the source property to the destination property.
    """
    source_declaration = ParticlePropertyDeclaration("source", np.dtype(dtype).type)
    destination_declaration = ParticlePropertyDeclaration("destination", np.dtype(dtype).type)

    kernel = ParticleKernel(
        add_property,
        particle_properties=[
            STATUS_DECLARATION,
            source_declaration,
            destination_declaration,
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "source": source,
            "destination": destination,
        },
    )


def construct_subtract_property_kernel(
    source: str,
    destination: str,
    dtype: SupportedDTypes = np.float64,
) -> BoundKernel:
    """Construct a kernel to subtract particle property source from destination.

    Parameters
    ----------
    source : str
        Binding for the source particle property.
    destination : str
        Binding for the destination particle property.
    dtype : type[np.float32] | type[np.float64] | type[np.integer], optional
        Data type of the particle properties (default np.float64).
        Supported types are np.float32, np.float64, and any np.integer.

    Returns
    -------
    BoundKernel
        A bound kernel that subtracts the source property from the destination property.
    """
    source_declaration = ParticlePropertyDeclaration("source", np.dtype(dtype).type)
    destination_declaration = ParticlePropertyDeclaration("destination", np.dtype(dtype).type)

    kernel = ParticleKernel(
        subtract_property,
        particle_properties=[
            STATUS_DECLARATION,
            source_declaration,
            destination_declaration,
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "source": source,
            "destination": destination,
        },
    )


def construct_multiply_property_kernel(
    source: str,
    destination: str,
    dtype: SupportedDTypes = np.float64,
) -> BoundKernel:
    """Construct a kernel to multiply particle property destination by source.

    Parameters
    ----------
    source : str
        Binding for the source particle property.
    destination : str
        Binding for the destination particle property.
    dtype : type[np.float32] | type[np.float64] | type[np.integer], optional
        Data type of the particle properties (default np.float64).
        Supported types are np.float32, np.float64, and any np.integer.

    Returns
    -------
    BoundKernel
        A bound kernel that multiplies the destination property by the source property.
    """
    source_declaration = ParticlePropertyDeclaration("source", np.dtype(dtype).type)
    destination_declaration = ParticlePropertyDeclaration("destination", np.dtype(dtype).type)

    kernel = ParticleKernel(
        multiply_property,
        particle_properties=[
            STATUS_DECLARATION,
            source_declaration,
            destination_declaration,
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "source": source,
            "destination": destination,
        },
    )


def construct_divide_property_kernel(
    source: str,
    destination: str,
    dtype: type[np.float32] | type[np.float64] = np.float64,
) -> BoundKernel:
    """Construct a kernel to divide particle property destination by source.

    Parameters
    ----------
    source : str
        Binding for the source particle property.
    destination : str
        Binding for the destination particle property.
    dtype : type[np.float32] | type[np.float64], optional
        Data type of the particle properties (default np.float64).
        Only np.float32 and np.float64 are supported.

    Returns
    -------
    BoundKernel
        A bound kernel that divides the destination property by the source property.
    """
    source_declaration = ParticlePropertyDeclaration("source", np.dtype(dtype).type)
    destination_declaration = ParticlePropertyDeclaration("destination", np.dtype(dtype).type)

    kernel = ParticleKernel(
        divide_property,
        particle_properties=[
            STATUS_DECLARATION,
            source_declaration,
            destination_declaration,
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "source": source,
            "destination": destination,
        },
    )
