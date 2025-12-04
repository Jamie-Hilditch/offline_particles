"""Advection scheme froms ROMS data with vertical velocity w.

Horizontal advection is done in index space. However, ROMS uses a sigma
coordinate system in the vertical, so vertical advection requires special treatment.
This module implements vertical advection in physical space using the vertical
velocity w and then transforms that into index space.
"""

import functools

import numpy.typing as npt

from ..interpolation import (
    bilinear_interpolation,
    linear_interpolation,
    trilinear_interpolation,
)
from ..kernel_data import KernelData
from ..kernel_tools import offset_indices, unwrap_scalar
from ..particle_kernel import ParticleKernel
from ..timesteppers import RK2Timestepper
from .vertical_coordinate import (
    S_coordinate,
    S_from_z,
    compute_zidx_from_S,
    sigma_coordinate,
    z_from_S,
)


def _rk2_step_1(
    particle: npt.NDArray,
    u: KernelData,
    v: KernelData,
    w: KernelData,
    dx: KernelData,
    dy: KernelData,
    hc: KernelData,
    NZ: KernelData,
    h: KernelData,
    C: KernelData,
    zeta: KernelData,
) -> None:
    """First rk2 step."""
    # get particle indices
    particle_indices = (
        particle["zidx"],
        particle["yidx"],
        particle["xidx"],
    )

    # compute z at current time 
    hc = unwrap_scalar(hc)
    NZ = unwrap_scalar(NZ)
    h_idx = offset_indices(particle_indices, h)
    C_idx = offset_indices(particle_indices, C)
    zeta_idx = offset_indices(particle_indices, zeta)   
    h_value = bilinear_interpolation(h_idx[0], h_idx[1], h.array)
    C_value = linear_interpolation(C_idx[0], C.array)
    zeta_value = bilinear_interpolation(zeta_idx[0], zeta_idx[1], zeta.array)

    sigma = sigma_coordinate(particle["zidx"], NZ)
    S = S_coordinate(hc, sigma, h_value, C_value)
    particle["z"] = z_from_S(S, h_value, zeta_value)

    # horizontal advection in index space
    # offset indices for u, v, dx, dy
    u_idx = offset_indices(particle_indices, u)
    v_idx = offset_indices(particle_indices, v)
    dx_idx = offset_indices(particle_indices, dx)
    dy_idx = offset_indices(particle_indices, dy)

    # interpolate u, v, dx, dy onto particle
    u_interp = trilinear_interpolation(u_idx[0], u_idx[1], u_idx[2], u.array)
    v_interp = trilinear_interpolation(v_idx[0], v_idx[1], v_idx[2], v.array)
    dx_interp = bilinear_interpolation(dx_idx[0], dx_idx[1], dx.array)
    dy_interp = bilinear_interpolation(dy_idx[0], dy_idx[1], dy.array)

    # compute rate of change of indices
    particle["_dxidx1"] = u_interp / dx_interp
    particle["_dyidx1"] = v_interp / dy_interp

    # vertical advection in physical space
    w_idx = offset_indices(particle_indices, w)
    w_interp = trilinear_interpolation(w_idx[0], w_idx[1], w_idx[2], w.array)
    particle["_dz1"] = w_interp

    

rk2_step_1_kernel = ParticleKernel(
    _rk2_step_1,
    particle_fields={
        "zidx": float, 
        "yidx": float, 
        "xidx": float, 
        "z": float, 
        "_dxidx1": float, 
        "_dyidx1": float, 
        "_dz1" :float
    },
    simulation_fields=[
        "u",
        "v",
        "w",
        "dx",
        "dy",
        "hc",
        "NZ",
        "h",
        "C",
        "zeta",
    ],
)


def _rk2_step_2(
    particle: npt.NDArray,
    dt: KernelData,
    alpha: KernelData,
    u: KernelData,
    v: KernelData,
    w: KernelData,
    dx: KernelData,
    dy: KernelData,
    hc: KernelData,
    NZ: KernelData,
    h: KernelData,
    C: KernelData,
    zeta: KernelData,
) -> None:
    """Second rk2 step."""
    dt = unwrap_scalar(dt)
    alpha = unwrap_scalar(alpha)

    # intermediate z position 
    z = particle["z"] + particle["_dz1"] * dt * alpha

    # intermediate xidx, yidx positions
    xidx = particle["xidx"] + particle["_dxidx1"] * dt * alpha
    yidx = particle["yidx"] + particle["_dyidx1"] * dt * alpha

    # intermediate zidx value 
    hc = unwrap_scalar(hc)
    NZ = unwrap_scalar(NZ)
    h_idx = offset_indices((0.0, yidx, xidx), h)
    zeta_idx = offset_indices((0.0, yidx, xidx), zeta)
    h_value = bilinear_interpolation(h_idx[0], h_idx[1], h.array)
    zeta_value = bilinear_interpolation(zeta_idx[0], zeta_idx[1], zeta.array)
    
    S = S_from_z(z, h_value, zeta_value)
    zidx = compute_zidx_from_S(S, hc, NZ, h_value, zeta_value, C)
    
    # particle indices at intermediate time
    particle_indices = (zidx, yidx, xidx)

    # horizontal advection in index space
    # offset indices for u, v, dx, dy
    u_idx = offset_indices(particle_indices, u)
    v_idx = offset_indices(particle_indices, v)
    dx_idx = offset_indices(particle_indices, dx)
    dy_idx = offset_indices(particle_indices, dy)

    # interpolate u, v, dx, dy onto particle
    u_interp = trilinear_interpolation(u_idx[0], u_idx[1], u_idx[2], u.array)
    v_interp = trilinear_interpolation(v_idx[0], v_idx[1], v_idx[2], v.array)
    dx_interp = bilinear_interpolation(dx_idx[0], dx_idx[1], dx.array)
    dy_interp = bilinear_interpolation(dy_idx[0], dy_idx[1], dy.array)

    # compute rate of change of indices
    particle["_dxidx2"] = u_interp / dx_interp
    particle["_dyidx2"] = v_interp / dy_interp

    # vertical advection in physical space
    w_idx = offset_indices(particle_indices, w)
    w_interp = trilinear_interpolation(w_idx[0], w_idx[1], w_idx[2], w.array)
    particle["_dz2"] = w_interp

rk2_step_2_kernel = ParticleKernel(
    _rk2_step_2,
    particle_fields={
        "zidx": float, 
        "yidx": float,
        "xidx": float,
        "z": float,
        "_dxidx2": float,
        "_dyidx2": float,
        "_dz2": float,
    },
    simulation_fields=[
        "_dt",
        "_RK2_alpha",
        "u",
        "v",
        "w",
        "dx",
        "dy",
        "hc",
        "NZ",
        "h",
        "C",
        "zeta",
    ],
)

def _rk2_update(
    particle: npt.NDArray,
    dt: KernelData,
    alpha: KernelData,
    hc: KernelData,
    NZ: KernelData,
    h: KernelData,
    C: KernelData,
    zeta: KernelData,
) -> None:
    """RK2 update step."""
    dt = unwrap_scalar(dt)
    alpha = unwrap_scalar(alpha)
    b1 = 1.0 - 1.0 / (2.0 * alpha)
    b2 = 1.0 / (2.0 * alpha)

    # update horizontal indices
    particle["xidx"] += b1 * dt * particle["_dxidx1"] + b2 * dt * particle["_dxidx2"]
    particle["yidx"] += b1 * dt * particle["_dyidx1"] + b2 * dt * particle["_dyidx2"]

    # update vertical position
    particle["z"] = particle["z"] + b1 * dt * particle["_dz1"] + b2 * dt * particle["_dz2"]

    # compute new zidx from updated z
    hc = unwrap_scalar(hc)
    NZ = unwrap_scalar(NZ)
    h_idx = offset_indices((0.0, particle["yidx"], particle["xidx"]), h)
    zeta_idx = offset_indices((0.0, particle["yidx"], particle["xidx"]), zeta)
    h_value = bilinear_interpolation(h_idx[0], h_idx[1], h.array)
    zeta_value = bilinear_interpolation(zeta_idx[0], zeta_idx[1], zeta.array)
    
    S = S_from_z(particle["z"], h_value, zeta_value)
    particle["zidx"] = compute_zidx_from_S(S, hc, NZ, h_value, zeta_value, C)

rk2_update_kernel = ParticleKernel(
    _rk2_update,
    particle_fields={
        "zidx": float,
        "yidx": float,
        "xidx": float,
        "z": float,
        "_dxidx1": float,
        "_dyidx1": float,
        "_dz1": float,
        "_dxidx2": float,
        "_dyidx2": float,
        "_dz2": float
    },
    simulation_fields=[
        "_dt",
        "_RK2_alpha",
        "hc",
        "NZ",
        "h",
        "C",
        "zeta",
    ],
)

"""Create an RK2 timesteppers with the ROMS w advection kernels."""
rk2_timestepper = functools.partial(
    RK2Timestepper,
    rk_step_1_kernel=rk2_step_1_kernel,
    rk_step_2_kernel=rk2_step_2_kernel,
    rk_update_kernel=rk2_update_kernel,
)
