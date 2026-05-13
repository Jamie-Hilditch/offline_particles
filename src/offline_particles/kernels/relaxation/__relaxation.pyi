"""Kernel functions for applying damping and relaxation to particle properties."""

from .._kernels import KernelFunction

linear_damping_accumulation: KernelFunction
quadratic_damping_accumulation: KernelFunction
linear_relaxation_accumulation: KernelFunction
quadratic_relaxation_accumulation: KernelFunction
