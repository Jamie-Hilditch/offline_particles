"""Validation kernels for offline particles."""

from ...spatial_arrays import BBox
from .._kernels import BoundKernel, ParticleKernel, ScalarDeclaration
from ..input_declarations import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from ._domain_bounds import construct_domain_bounds_kernel
from ._finite_indices import finite_indices_kernel

__all__ = [
    "STATUS_DECLARATION",
    "XIDX_DECLARATION",
    "YIDX_DECLARATION",
    "ZIDX_DECLARATION",
    "ParticleKernel",
    "ScalarDeclaration",
    "construct_domain_bounds_kernel",
    "construct_validation_kernel",
    "finite_indices_kernel",
]


def construct_validation_kernel(
    zmin: float,
    zmax: float,
    ymin: float,
    ymax: float,
    xmin: float,
    xmax: float,
) -> BoundKernel:
    """Construct a bound kernel that checks that particles have finite indices and remain in the domain.

    Parameters
    ----------
    zmin : float
        Minimum z index of the domain.
    zmax : float
        Maximum z index of the domain.
    ymin : float
        Minimum y index of the domain.
    ymax : float
        Maximum y index of the domain.
    xmin : float
        Minimum x index of the domain.
    xmax : float
        Maximum x index of the domain.

    Returns
    -------
    BoundKernel
        A bound kernel that checks that particles have finite indices and remain in the domain.
    """
    domain_bounds_kernel = construct_domain_bounds_kernel(zmin, zmax, ymin, ymax, xmin, xmax)
    return BoundKernel.chain(finite_indices_kernel, domain_bounds_kernel)


def construct_validation_kernel_from_bbox(bbox: BBox) -> BoundKernel:
    """Construct a bound kernel that checks that particles have finite indices and remain in the domain.

    Parameters
    ----------
    bbox : BBox
        The bounding box defining the domain.

    Returns
    -------
    BoundKernel
        A bound kernel that checks that particles have finite indices and remain in the domain.
    """
    return construct_validation_kernel(
        zmin=bbox.zmin,
        zmax=bbox.zmax,
        ymin=bbox.ymin,
        ymax=bbox.ymax,
        xmin=bbox.xmin,
        xmax=bbox.xmax,
    )
