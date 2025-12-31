from typing import Literal

from .._kernels import KernelFunction, ParticleKernel

linear_interpolation_kernel_function: KernelFunction
bilinear_interpolation_kernel_function: KernelFunction
trilinear_interpolation_kernel_function: KernelFunction

def linear_interpolation_kernel(
    field: str, output_name: str | None = None, dimension: Literal["z", "y", "x"] = "z"
) -> ParticleKernel: ...
def bilinear_interpolation_kernel(
    field: str, output_name: str | None = None, dimensions: tuple[str, str] = ("y", "x")
) -> ParticleKernel: ...
def trilinear_interpolation_kernel(field: str, output_name: str | None = None) -> ParticleKernel: ...
def vertical_interpolation_kernel(field: str, output_name: str | None = None) -> ParticleKernel: ...
def horizontal_interpolation_kernel(field: str, output_name: str | None = None) -> ParticleKernel: ...
