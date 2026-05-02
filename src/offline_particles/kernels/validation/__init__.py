import numpy as np

from .._kernels import BoundKernel, ParticleKernel, ScalarDeclaration
from ..input_declarations import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from ._validation import domain_bounds, finite_indices

__all__ = [
    "construct_domain_bounds_kernel",
    "construct_finite_indices_kernel",
    "construct_validation_kernel",
    "domain_bounds_kernel",
    "finite_indices_kernel",
    "validation_kernel",
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


# bound kernels
def construct_finite_indices_kernel() -> BoundKernel:
    """Construct the finite indices validation bound kernel.

    Returns:
        BoundKernel implementing the finite indices validation.
    """
    return finite_indices_kernel.bind()


def construct_domain_bounds_kernel(
    zidx_min: str = "zidx_min",
    zidx_max: str = "zidx_max",
    yidx_min: str = "yidx_min",
    yidx_max: str = "yidx_max",
    xidx_min: str = "xidx_min",
    xidx_max: str = "xidx_max",
) -> BoundKernel:
    """Construct the domain bounds validation bound kernel.

    Args:
        zidx_min: Binding for the minimum valid z-index scalar (default "zidx_min").
        zidx_max: Binding for the maximum valid z-index scalar (default "zidx_max").
        yidx_min: Binding for the minimum valid y-index scalar (default "yidx_min").
        yidx_max: Binding for the maximum valid y-index scalar (default "yidx_max").
        xidx_min: Binding for the minimum valid x-index scalar (default "xidx_min").
        xidx_max: Binding for the maximum valid x-index scalar (default "xidx_max").

    Returns:
        BoundKernel implementing the domain bounds validation.
    """
    return domain_bounds_kernel.bind(
        scalars={
            "zidx_min": zidx_min,
            "zidx_max": zidx_max,
            "yidx_min": yidx_min,
            "yidx_max": yidx_max,
            "xidx_min": xidx_min,
            "xidx_max": xidx_max,
        }
    )


def construct_validation_kernel(
    zidx_min: str = "zidx_min",
    zidx_max: str = "zidx_max",
    yidx_min: str = "yidx_min",
    yidx_max: str = "yidx_max",
    xidx_min: str = "xidx_min",
    xidx_max: str = "xidx_max",
) -> BoundKernel:
    """Construct the full validation bound kernel.

    Args:
        zidx_min: Binding for the minimum valid z-index scalar (default "zidx_min").
        zidx_max: Binding for the maximum valid z-index scalar (default "zidx_max").
        yidx_min: Binding for the minimum valid y-index scalar (default "yidx_min").
        yidx_max: Binding for the maximum valid y-index scalar (default "yidx_max").
        xidx_min: Binding for the minimum valid x-index scalar (default "xidx_min").
        xidx_max: Binding for the maximum valid x-index scalar (default "xidx_max").

    Returns:
        BoundKernel implementing particle validation.
    """
    return validation_kernel.bind(
        scalars={
            "zidx_min": zidx_min,
            "zidx_max": zidx_max,
            "yidx_min": yidx_min,
            "yidx_max": yidx_max,
            "xidx_min": xidx_min,
            "xidx_max": xidx_max,
        }
    )
