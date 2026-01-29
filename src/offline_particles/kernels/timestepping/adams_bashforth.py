"""ParticleKernels for Adams-Bashforth timestepping schemes."""

import numpy as np
import numpy.typing as npt

from .._kernels import BoundKernel, ParticleKernel, ParticlePropertyDeclaration
from ..common_inputs import DT_DECLARATION, STATUS_DECLARATION
from ._adams_bashforth import (
    ab2_bump_status,
    ab2_initialisation,
    ab2_update_float32,
    ab2_update_float64,
    ab3_bump_status,
    ab3_initialisation,
    ab3_update_float32,
    ab3_update_float64,
)


# particle property declarations for Adams-Bashforth
def _particle_property_declarations(dtype: type[np.float32] | type[np.float64]) -> list[ParticlePropertyDeclaration]:
    """Helper function to create particle property declarations for Adams-Bashforth kernels.

    Args:
        dtype: Data type of the particle properties (np.float32 or np.float64).
    """
    prop_declaration = ParticlePropertyDeclaration("prop", np.dtype(dtype))
    dprop_0_declaration = ParticlePropertyDeclaration("dprop_0", np.dtype(dtype))
    dprop_1_declaration = ParticlePropertyDeclaration("dprop_1", np.dtype(dtype))
    dprop_2_declaration = ParticlePropertyDeclaration("dprop_2", np.dtype(dtype))
    return [
        STATUS_DECLARATION,
        prop_declaration,
        dprop_0_declaration,
        dprop_1_declaration,
        dprop_2_declaration,
    ]


def construct_ab2_update_kernel(
    prop: str, dprop_0: str, dprop_1: str, dtype: npt.DTypeLike = np.float32
) -> BoundKernel:
    """Construct an Adams-Bashforth 2 update kernel for a given property.

    Args:
        prop: Binding of the property to be updated.
        dprop_0: Binding of the property tendency at the current timestep.
        dprop_1: Binding of the property tendency at the previous timestep.
        dtype: Data type of the particle properties (np.float32 or np.float64).

    Returns:
        BoundKernel implementing the AB2 update.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        ab2_update = ab2_update_float32
    elif dtype == np.float64:
        ab2_update = ab2_update_float64
    else:
        raise ValueError("Unsupported dtype for Adams-Bashforth 2 kernel. Use np.float32 or np.float64.")
    kernel = ParticleKernel(
        ab2_update,
        particle_properties=_particle_property_declarations(dtype.type)[:4],
        scalars=[DT_DECLARATION],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "prop": prop,
            "dprop_0": dprop_0,
            "dprop_1": dprop_1,
        },
    )


def construct_ab2_bump_status_kernel() -> BoundKernel:
    """Construct an Adams-Bashforth 2 bump status kernel.

    Returns:
        BoundKernel implementing the AB2 bump status.
    """

    kernel = ParticleKernel(
        ab2_bump_status,
        particle_properties=[STATUS_DECLARATION],
    )
    return BoundKernel(kernel)


def construct_ab2_initialisation_kernel() -> BoundKernel:
    """Construct an Adams-Bashforth 2 initialisation kernel.

    Returns:
        BoundKernel implementing the AB2 initialisation.
    """

    kernel = ParticleKernel(
        ab2_initialisation,
        particle_properties=[
            STATUS_DECLARATION,
        ],
    )
    return BoundKernel(kernel)


def construct_ab3_update_kernel(
    prop: str, dprop_0: str, dprop_1: str, dprop_2: str, dtype: npt.DTypeLike = np.float64
) -> BoundKernel:
    """Construct an Adams-Bashforth 3 update kernel for a given property.

    Args:
        prop: Binding of the property to be updated.
        dprop_0: Binding of the property tendency at the current timestep.
        dprop_1: Binding of the property tendency at the previous timestep.
        dprop_2: Binding of the property tendency at two timesteps ago.
        dtype: Data type of the particle properties (np.float32 or np.float64).

    Returns:
        BoundKernel implementing the AB3 update.
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        ab3_update = ab3_update_float32
    elif dtype == np.float64:
        ab3_update = ab3_update_float64
    kernel = ParticleKernel(
        ab3_update,
        particle_properties=_particle_property_declarations(dtype.type),
        scalars=[DT_DECLARATION],
    )
    return BoundKernel(
        kernel,
        particle_property_bindings={
            "prop": prop,
            "dprop_0": dprop_0,
            "dprop_1": dprop_1,
            "dprop_2": dprop_2,
        },
    )


def construct_ab3_bump_status_kernel() -> BoundKernel:
    """Construct an Adams-Bashforth 3 bump status kernel.

    Returns:
        BoundKernel implementing the AB3 bump status.
    """

    kernel = ParticleKernel(
        ab3_bump_status,
        particle_properties=[STATUS_DECLARATION],
    )
    return BoundKernel(kernel)


def construct_ab3_initialisation_kernel() -> BoundKernel:
    """Construct an Adams-Bashforth 3 initialisation kernel.

    Returns:
        BoundKernel implementing the AB3 initialisation.
    """

    kernel = ParticleKernel(
        ab3_initialisation,
        particle_properties=[
            STATUS_DECLARATION,
        ],
    )
    return BoundKernel(kernel)
