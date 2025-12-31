from typing import Literal

import numpy as np

from ...fields import FieldData
from ...particles import Particles
from .._kernels import ParticleKernel

def linear_interpolation_kernel_function(
    particles: Particles,
    scalars: dict[str, np.number],
    fielddata: dict[str, FieldData],
    dimension_idx: str,
    field_name: str,
    particle_name: str,
) -> None: ...
def bilinear_interpolation_kernel_function(
    particles: Particles,
    scalars: dict[str, np.number],
    fielddata: dict[str, FieldData],
    dimension_idx0: str,
    dimension_idx1: str,
    field_name: str,
    particle_name: str,
) -> None: ...
def trilinear_interpolation_kernel_function(
    particles: Particles,
    scalars: dict[str, np.number],
    fielddata: dict[str, FieldData],
    field_name: str,
    particle_name: str,
) -> None: ...
def linear_interpolation_kernel(
    field: str, output_name: str | None = None, dimension: Literal["z", "y", "x"] = "z"
) -> ParticleKernel: ...
def bilinear_interpolation_kernel(
    field: str, output_name: str | None = None, dimensions: tuple[str, str] = ("y", "x")
) -> ParticleKernel: ...
def trilinear_interpolation_kernel(field: str, output_name: str | None = None) -> ParticleKernel: ...
def vertical_interpolation_kernel(field: str, output_name: str | None = None) -> ParticleKernel: ...
def horizontal_interpolation_kernel(field: str, output_name: str | None = None) -> ParticleKernel: ...
