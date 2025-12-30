from typing import Callable

from ...timesteppers import RK2Timestepper
from .._kernels import KernelFunction, ParticleKernel

rk2_w_advection_step_1: KernelFunction
rk2_w_advection_step_2: KernelFunction
rk2_w_advection_update: KernelFunction
rk2_w_advection_step_1_kernel: ParticleKernel
rk2_w_advection_step_2_kernel: ParticleKernel
rk2_w_advection_update_kernel: ParticleKernel
rk2_w_advection_timestepper: Callable[..., RK2Timestepper]
