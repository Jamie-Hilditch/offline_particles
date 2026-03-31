"""Kernels for timed activation, i.e. releases and retirement, of particles."""

import numpy as np

from .._kernels import BoundKernel, ParticleKernel, ParticlePropertyDeclaration
from ..common_inputs import STATUS_DECLARATION, construct_time_declaration
from ._timed_activation import (
    activate_released_particles,
    deactivate_retired_particles,
)

__all__ = [
    "construct_activate_released_particles_kernel",
    "construct_deactivate_retired_particles_kernel",
]

type SupportedDTypes = type[np.float64] | type[np.datetime64]


def construct_activate_released_particles_kernel(
    release_time: str = "release_time", dtype: SupportedDTypes = np.float64
) -> BoundKernel:
    """Construct a kernel to activate particles at a given release time.

    Args:
        release_time: Binding for the release time particle property (default "release_time").
        dtype: Data type of both the simulation time and the release_time particle property (default np.float64).
            Supported types are np.float64 and np.datetime64.
    """
    release_time_declaration = ParticlePropertyDeclaration("release_time", np.dtype(dtype))

    kernel = ParticleKernel(
        activate_released_particles,
        particle_properties=[
            STATUS_DECLARATION,
            release_time_declaration,
        ],
        scalars=[
            construct_time_declaration(dtype),
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "release_time": release_time,
        },
    )


def construct_deactivate_retired_particles_kernel(
    retirement_time: str = "retirement_time", dtype: SupportedDTypes = np.float64
) -> BoundKernel:
    """Construct a kernel to deactivate particles at a given retirement time.

    Args:
        retirement_time: Binding for the retirement time particle property (default "retirement_time").
        dtype: Data type of both the simulation time and the retirement_time particle property (default np.float64).
            Supported types are np.float64 and np.datetime64.
    """
    retirement_time_declaration = ParticlePropertyDeclaration("retirement_time", np.dtype(dtype))

    kernel = ParticleKernel(
        deactivate_retired_particles,
        particle_properties=[
            STATUS_DECLARATION,
            retirement_time_declaration,
        ],
        scalars=[
            construct_time_declaration(dtype),
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "retirement_time": retirement_time,
        },
    )
