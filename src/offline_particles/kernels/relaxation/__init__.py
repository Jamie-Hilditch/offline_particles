"""Kernels for applying relaxation and damping to particle properties."""

import numpy as np

from ..._kernels import KernelBinding, ParticleKernel, ParticlePropertyDeclaration, ScalarDeclaration
from ...common_inputs import STATUS_DECLARATION
from ._relaxation import (
    linear_damping_accumulation,
    linear_relaxation_accumulation,
    quadratic_damping_accumulation,
    quadratic_relaxation_accumulation,
)

__all__ = [
    "construct_linear_damping_accumulation_kernel",
    "construct_quadratic_damping_accumulation_kernel",
    "construct_linear_relaxation_accumulation_kernel",
    "construct_quadratic_relaxation_accumulation_kernel",
]

prop_declaration = ParticlePropertyDeclaration("prop", np.float64)
rhs_declaration = ParticlePropertyDeclaration("rhs", np.float64)
target_declaration = ParticlePropertyDeclaration("target", np.float64)
linear_damping_declaration = ScalarDeclaration("linear_damping_coefficient", np.float64)
quadratic_damping_declaration = ScalarDeclaration("quadratic_damping_coefficient", np.float64)
linear_relaxation_declaration = ScalarDeclaration("linear_relaxation_coefficient", np.float64)
quadratic_relaxation_declaration = ScalarDeclaration("quadratic_relaxation_coefficient", np.float64)


def construct_linear_damping_accumulation_kernel(
    rhs: str,
    prop: str,
    coefficient: str,
) -> KernelBinding:
    """Construct a kernel to apply linear damping to a particle property."""
    kernel = ParticleKernel(
        linear_damping_accumulation,
        particle_properties=[
            STATUS_DECLARATION,
            prop_declaration,
            rhs_declaration,
        ],
        scalars=[
            linear_damping_declaration,
        ],
    )
    return KernelBinding(
        kernel,
        particle_property_bindings={
            "rhs": rhs,
            "prop": prop,
        },
        scalar_bindings={
            "linear_damping_coefficient": coefficient,
        },
    )


def construct_quadratic_damping_accumulation_kernel(
    rhs: str,
    prop: str,
    coefficient: str,
) -> KernelBinding:
    """Construct a kernel to apply quadratic damping to a particle property."""
    kernel = ParticleKernel(
        quadratic_damping_accumulation,
        particle_properties=[
            STATUS_DECLARATION,
            prop_declaration,
            rhs_declaration,
        ],
        scalars=[
            quadratic_damping_declaration,
        ],
    )
    return KernelBinding(
        kernel,
        particle_property_bindings={
            "rhs": rhs,
            "prop": prop,
        },
        scalar_bindings={
            "quadratic_damping_coefficient": coefficient,
        },
    )


def construct_linear_relaxation_accumulation_kernel(
    rhs: str,
    prop: str,
    target: str,
    coefficient: str,
) -> KernelBinding:
    """Construct a kernel to apply linear relaxation to a particle property."""
    kernel = ParticleKernel(
        linear_relaxation_accumulation,
        particle_properties=[
            STATUS_DECLARATION,
            prop_declaration,
            target_declaration,
            rhs_declaration,
        ],
        scalars=[
            linear_relaxation_declaration,
        ],
    )
    return KernelBinding(
        kernel,
        particle_property_bindings={
            "rhs": rhs,
            "prop": prop,
            "target": target,
        },
        scalar_bindings={
            "linear_relaxation_coefficient": coefficient,
        },
    )


def construct_quadratic_relaxation_accumulation_kernel(
    rhs: str,
    prop: str,
    target: str,
    coefficient: str,
) -> KernelBinding:
    """Construct a kernel to apply quadratic relaxation to a particle property."""
    kernel = ParticleKernel(
        quadratic_relaxation_accumulation,
        particle_properties=[
            STATUS_DECLARATION,
            prop_declaration,
            target_declaration,
            rhs_declaration,
        ],
        scalars=[
            quadratic_relaxation_declaration,
        ],
    )
    return KernelBinding(
        kernel,
        particle_property_bindings={
            "rhs": rhs,
            "prop": prop,
            "target": target,
        },
        scalar_bindings={
            "quadratic_relaxation_coefficient": coefficient,
        },
    )
