import numpy as np

from .._kernels import ParticleKernel, ScalarDeclaration
from ..common_inputs import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from ._validation import domain_bounds, finite_indices

__all__ = [
    "domain_bounds_kernel",
    "domain_bounds_kernel_binding",
    "finite_indices_kernel",
    "finite_indices_kernel_binding",
    "validation_kernel",
    "validation_kernel_binding",
]

finite_indices_kernel = ParticleKernel(
    finite_indices,
    particle_properties=[
        STATUS_DECLARATION,
        ZIDX_DECLARATION,
        YIDX_DECLARATION,
        XIDX_DECLARATION,
    ],
)
domain_bounds_kernel = ParticleKernel(
    domain_bounds,
    particle_properties=[
        STATUS_DECLARATION,
        ZIDX_DECLARATION,
        YIDX_DECLARATION,
        XIDX_DECLARATION,
    ],
    scalars=[
        ScalarDeclaration("zidx_min", np.float64),
        ScalarDeclaration("zidx_max", np.float64),
        ScalarDeclaration("yidx_min", np.float64),
        ScalarDeclaration("yidx_max", np.float64),
        ScalarDeclaration("xidx_min", np.float64),
        ScalarDeclaration("xidx_max", np.float64),
    ],
)
validation_kernel = ParticleKernel.chain(finite_indices_kernel, domain_bounds_kernel)

# kernel bindings
finite_indices_kernel_binding = finite_indices_kernel.bind()
domain_bounds_kernel_binding = domain_bounds_kernel.bind()
validation_kernel_binding = validation_kernel.bind()
