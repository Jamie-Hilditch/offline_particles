"""Kernel to check that particles remain in the domain."""

import numba
import numpy as np
import numpy.typing as npt

from .._kernels import BoundKernel, FieldDataType, ParticleKernel, ParticlePropertiesType, ScalarsType
from ..input_declarations import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from ..status import INACTIVE_FLAG, Status

__all__ = ["construct_domain_bounds_kernel"]

_BELOW_BOTTOM_STATUS = np.uint8(Status.BELOW_BOTTOM)
_ABOVE_SURFACE_STATUS = np.uint8(Status.ABOVE_SURFACE)
_OUT_OF_DOMAIN_STATUS = np.uint8(Status.OUT_OF_DOMAIN)


def construct_domain_bounds_kernel(
    zmin: float, zmax: float, ymin: float, ymax: float, xmin: float, xmax: float
) -> BoundKernel:
    """Construct a bound kernel to check that particles remain in the domain.

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
        A bound kernel that checks that particles remain in the domain.

    Raises
    ------
    ValueError
        If any of the minimum bounds are greater than the corresponding maximum bounds.
    """
    # first some basic validation
    if zmin > zmax:
        raise ValueError(f"Minimum z index (zmin={zmin}) cannot be greater than maximum z index (zmax={zmax}).")
    if ymin > ymax:
        raise ValueError(f"Minimum y index (ymin={ymin}) cannot be greater than maximum y index (ymax={ymax}).")
    if xmin > xmax:
        raise ValueError(f"Minimum x index (xmin={xmin}) cannot be greater than maximum x index (xmax={xmax}).")

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def _domain_bounds(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
    ) -> None:
        for i in numba.prange(status.size):  # type: ignore[not-iterable]
            if status[i] & INACTIVE_FLAG:
                continue

            if zidx[i] < zmin:
                status[i] = _BELOW_BOTTOM_STATUS
            elif zidx[i] > zmax:
                status[i] = _ABOVE_SURFACE_STATUS

            # if any index is out of bounds mark as invalid
            # note this takes precedence over vertical checks
            if not (yidx[i] >= ymin and yidx[i] <= ymax and xidx[i] >= xmin and xidx[i] <= xmax):
                status[i] = _OUT_OF_DOMAIN_STATUS

    def _domain_bounds_kernel_function(
        particle_properties: ParticlePropertiesType,
        scalars: ScalarsType,
        field_data: FieldDataType,
    ) -> None:
        _domain_bounds(
            particle_properties["status"],
            particle_properties["zidx"],
            particle_properties["yidx"],
            particle_properties["xidx"],
        )

    kernel = ParticleKernel(
        _domain_bounds_kernel_function,
        particle_properties=[
            STATUS_DECLARATION,
            ZIDX_DECLARATION,
            YIDX_DECLARATION,
            XIDX_DECLARATION,
        ],
    )

    bound_kernel = kernel.bind()
    return bound_kernel
