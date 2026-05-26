"""Create a kernel to validate that particle indices are finite."""

import numba
import numpy as np
import numpy.typing as npt

from .._kernels import BoundKernel, FieldDataType, ParticleKernel, ParticlePropertiesType, ScalarsType
from ..input_declarations import STATUS_DECLARATION, XIDX_DECLARATION, YIDX_DECLARATION, ZIDX_DECLARATION
from ..status import INACTIVE_FLAG, Status

__all__ = ["finite_indices_kernel"]

_NONFINITE_STATUS = np.uint8(Status.NONFINITE)


@numba.njit(parallel=True, nogil=True)
def _finite_indices(
    status: npt.NDArray[np.uint8],
    zidx: npt.NDArray[np.float64],
    yidx: npt.NDArray[np.float64],
    xidx: npt.NDArray[np.float64],
) -> None:
    """Kernel function implementation to check that particle indices are finite.

    Parameters
    ----------
    status
        The particle status array.
    zidx
        The particle z index array.
    yidx
        The particle y index array.
    xidx
        The particle x index array.
    """
    for i in numba.prange(status.size):  # type: ignore[not-iterable]
        if status[i] & INACTIVE_FLAG:
            continue

        if not np.isfinite(zidx[i]) or not np.isfinite(yidx[i]) or not np.isfinite(xidx[i]):
            status[i] = _NONFINITE_STATUS


def _finite_indices_kernel_function(
    particle_properties: ParticlePropertiesType,
    scalars: ScalarsType,
    field_data: FieldDataType,
) -> None:
    _finite_indices(
        particle_properties["status"],
        particle_properties["zidx"],
        particle_properties["yidx"],
        particle_properties["xidx"],
    )


finite_indices_kernel: BoundKernel = ParticleKernel(
    _finite_indices_kernel_function,
    particle_properties=[
        STATUS_DECLARATION,
        ZIDX_DECLARATION,
        YIDX_DECLARATION,
        XIDX_DECLARATION,
    ],
).bind()
