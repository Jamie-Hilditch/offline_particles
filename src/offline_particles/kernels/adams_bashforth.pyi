import numpy as np
import numpy.typing as npt

from ..fields import FieldData
from ..particles import Particles
from ._kernels import KernelFunction, ParticleKernel

def ab2_update(
    particles: Particles,
    scalars: dict[str, np.generic],
    fielddata: dict[str, FieldData],
    field: str,
    tendency_field_0: str,
    tendency_field_1: str,
) -> None: ...

ab2_bunp_status: KernelFunction

def ab3_update(
    particles: Particles,
    scalars: dict[str, np.generic],
    fielddata: dict[str, FieldData],
    field: str,
    tendency_field_0: str,
    tendency_field_1: str,
    tendency_field_2: str,
) -> None: ...

ab3_bump_status: KernelFunction

def ab2_update_kernel(
    field: str, tendency_field_0: str | None = None, tendency_field_1: str | None = None, dtype: npt.Dtype = np.float64
) -> ParticleKernel: ...

ab2_bump_status_kernel: ParticleKernel

def ab3_update_kernel(
    field: str,
    tendency_field_0: str | None = None,
    tendency_field_1: str | None = None,
    tendency_field_2: str | None = None,
    dtype: npt.Dtype = np.float64,
) -> ParticleKernel: ...

ab3_bump_status_kernel: ParticleKernel
