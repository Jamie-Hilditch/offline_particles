"""Kernels for timed activation, i.e. releases and retirement, of particles."""

import numpy as np
import numpy.typing as npt

from .._kernels import BoundKernel, ParticleKernel, ParticlePropertyDeclaration
from ..input_declarations import DT_DECLARATION, STATUS_DECLARATION, construct_time_declaration
from ._timed_activation import (
    activate_released_particles,
    deactivate_retired_particles,
)

__all__ = [
    "construct_activate_released_particles_kernel",
    "construct_deactivate_retired_particles_kernel",
]


def construct_activate_released_particles_kernel(
    release_time: str = "release_time", dtype: npt.DTypeLike = np.float64
) -> BoundKernel:
    """Construct a kernel to activate particles at a given release time.

    Parameters
    ----------
    release_time : str, optional
        Binding for the release time particle property (default "release_time").
    dtype : npt.DTypeLike, optional
        Data type of both the simulation time and the release_time particle property
        (default np.float64). Use np.float64 for float-based clocks. For datetime-based
        clocks, pass an explicit datetime64 dtype with a unit matching the simulation
        clock's time array, e.g. np.dtype('datetime64[ns]').

    Returns
    -------
    BoundKernel
        A bound kernel that activates particles at the given release time.
    """
    release_time_declaration = ParticlePropertyDeclaration("release_time", np.dtype(dtype).type)

    kernel = ParticleKernel(
        activate_released_particles,
        particle_properties=[
            STATUS_DECLARATION,
            release_time_declaration,
        ],
        scalars=[
            construct_time_declaration(dtype),
            DT_DECLARATION,
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "release_time": release_time,
        },
    )


def construct_deactivate_retired_particles_kernel(
    retirement_time: str = "retirement_time", dtype: npt.DTypeLike = np.float64
) -> BoundKernel:
    """Construct a kernel to deactivate particles at a given retirement time.

    Parameters
    ----------
    retirement_time : str, optional
        Binding for the retirement time particle property (default "retirement_time").
    dtype : npt.DTypeLike, optional
        Data type of both the simulation time and the retirement_time particle property
        (default np.float64). Use np.float64 for float-based clocks. For datetime-based
        clocks, pass an explicit datetime64 dtype with a unit matching the simulation
        clock's time array, e.g. np.dtype('datetime64[ns]').

    Returns
    -------
    BoundKernel
        A bound kernel that deactivates particles at the given retirement time.
    """
    retirement_time_declaration = ParticlePropertyDeclaration("retirement_time", np.dtype(dtype).type)

    kernel = ParticleKernel(
        deactivate_retired_particles,
        particle_properties=[
            STATUS_DECLARATION,
            retirement_time_declaration,
        ],
        scalars=[
            construct_time_declaration(dtype),
            DT_DECLARATION,
        ],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "retirement_time": retirement_time,
        },
    )
