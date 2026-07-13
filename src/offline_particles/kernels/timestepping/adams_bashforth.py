"""ParticleKernels for Adams-Bashforth timestepping schemes."""

import numpy as np
import numpy.typing as npt

from .._kernels import (
    BoundKernel,
    FieldDataType,
    ParticleKernel,
    ParticlePropertiesType,
    ParticlePropertyDeclaration,
    ScalarsType,
)
from ..input_declarations import DT_DECLARATION, STATUS_DECLARATION
from ..status import Status, construct_initialise_status_kernel
from ._adams_bashforth import ab2_update, ab3_update, ab_bump_status


# particle property declarations for Adams-Bashforth
def _particle_property_declarations(dtype: type[np.inexact], order: int) -> list[ParticlePropertyDeclaration]:
    """Create particle property declarations for Adams-Bashforth kernels.

    Parameters
    ----------
    dtype : np.inexact
        Data type of the particle properties.
    order : int
        Order of the Adams-Bashforth scheme.

    Returns
    -------
    list[ParticlePropertyDeclaration]
        List of particle property declarations for the property plus its current and previous tendencies.
    """
    prop_declaration = ParticlePropertyDeclaration("prop", dtype)
    dprop_0_declaration = ParticlePropertyDeclaration("dprop_0", dtype)
    dprop_1_declaration = ParticlePropertyDeclaration("dprop_1", dtype)
    dprop_2_declaration = ParticlePropertyDeclaration("dprop_2", dtype)

    declarations = [
        STATUS_DECLARATION,
        prop_declaration,
        dprop_0_declaration,
        dprop_1_declaration,
    ]

    if order > 2:
        declarations.append(dprop_2_declaration)

    return declarations


def construct_ab2_update_kernel(
    prop: str, dprop_0: str, dprop_1: str, dtype: npt.DTypeLike = np.float32
) -> BoundKernel:
    """Construct an Adams-Bashforth 2 update kernel for a given property.

    Parameters
    ----------
    prop : str
        Binding of the property to be updated.
    dprop_0 : str
        Binding of the property tendency at the current timestep.
    dprop_1 : str
        Binding of the property tendency at the previous timestep.
    dtype : npt.DTypeLike, optional
        Data type of the particle properties (np.float32 or np.float64).

    Returns
    -------
    BoundKernel
        BoundKernel implementing the AB2 update.
    """
    dtype = np.dtype(dtype)

    def kernel_function(
        particle_properties: ParticlePropertiesType,
        scalars: ScalarsType,
        fields: FieldDataType,
    ) -> None:
        """Adams-Bashforth 2 update kernel function to be used in the ParticleKernel."""
        prop = particle_properties["prop"]
        dprop_0 = particle_properties["dprop_0"]
        dprop_1 = particle_properties["dprop_1"]
        dt = scalars["_dt"]

        ab2_update(particle_properties["status"], prop, dprop_0, dprop_1, dt)

    kernel = ParticleKernel(
        kernel_function,
        particle_properties=_particle_property_declarations(dtype.type, 2),
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


def construct_ab3_update_kernel(
    prop: str, dprop_0: str, dprop_1: str, dprop_2: str, dtype: npt.DTypeLike = np.float64
) -> BoundKernel:
    """Construct an Adams-Bashforth 3 update kernel for a given property.

    Parameters
    ----------
    prop : str
        Binding of the property to be updated.
    dprop_0 : str
        Binding of the property tendency at the current timestep.
    dprop_1 : str
        Binding of the property tendency at the previous timestep.
    dprop_2 : str
        Binding of the property tendency at two timesteps ago.
    dtype : npt.DTypeLike, optional
        Data type of the particle properties (np.float32 or np.float64).

    Returns
    -------
    BoundKernel
        BoundKernel implementing the AB3 update.
    """
    dtype = np.dtype(dtype)

    def kernel_function(
        particle_properties: ParticlePropertiesType,
        scalars: ScalarsType,
        fields: FieldDataType,
    ) -> None:
        """Adams-Bashforth 3 update kernel function to be used in the ParticleKernel."""
        prop = particle_properties["prop"]
        dprop_0 = particle_properties["dprop_0"]
        dprop_1 = particle_properties["dprop_1"]
        dprop_2 = particle_properties["dprop_2"]
        dt = scalars["_dt"]

        ab3_update(particle_properties["status"], prop, dprop_0, dprop_1, dprop_2, dt)

    kernel = ParticleKernel(
        kernel_function,
        particle_properties=_particle_property_declarations(dtype.type, 3),
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


def construct_ab_bump_status_kernel() -> BoundKernel:
    """Construct a kernel that bumps the Adams-Bashforth status.

    Returns
    -------
    BoundKernel
        BoundKernel implementing AB bump status.
    """

    def kernel_function(
        particle_properties: ParticlePropertiesType,
        scalars: ScalarsType,
        fields: FieldDataType,
    ) -> None:
        """Adams-Bashforth bump status kernel function to be used in the ParticleKernel."""
        ab_bump_status(particle_properties["status"])

    kernel = ParticleKernel(
        kernel_function,
        particle_properties=[STATUS_DECLARATION],
    )
    return BoundKernel(kernel)


def ab_initial_status(order: int) -> Status:
    """Map an Adams-Bashforth order to the multistep status a particle should start initialisation at.

    Parameters
    ----------
    order : int
        The order of the Adams-Bashforth method (2 or 3).

    Returns
    -------
    Status
        ``Status.MULTISTEP_2`` for order 3, ``Status.MULTISTEP_1`` for order 2.

    Raises
    ------
    ValueError
        If an unsupported Adams-Bashforth order is specified.
    """
    if order == 2:
        return Status.MULTISTEP_1
    elif order == 3:
        return Status.MULTISTEP_2
    else:
        raise ValueError(f"Unsupported Adams-Bashforth order: {order}")


def construct_ab_initialisation_kernel(order: int) -> BoundKernel:
    """Construct a kernel that finalizes initialisation with the Adams-Bashforth startup status.

    Transitions particles with status ``Status.INITIALISING`` to the appropriate multistep status
    for the given Adams-Bashforth `order` (``Status.MULTISTEP_2`` for order 3,
    ``Status.MULTISTEP_1`` for order 2), so they get the correct startup ramp-up. Delegates to
    :func:`~offline_particles.kernels.status.construct_initialise_status_kernel`.

    Parameters
    ----------
    order : int
        The order of the Adams-Bashforth method (2 or 3).

    Returns
    -------
    BoundKernel
        BoundKernel implementing the AB initialisation.

    Raises
    ------
    ValueError
        If an unsupported Adams-Bashforth order is specified.
    """
    return construct_initialise_status_kernel(ab_initial_status(order))
