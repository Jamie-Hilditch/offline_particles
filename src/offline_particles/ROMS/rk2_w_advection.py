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
from ..kernel_tools import offset_indices_1D, offset_indices_2D, offset_indices_3D
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
    hc: float,
    NZ: int,
    u: npt.NDArray[float], 
    u_off: tuple[float, float, float],
    v: npt.NDArray[float],
    v_off: tuple[float, float, float],
    w: npt.NDArray[float],
    w_off: tuple[float, float, float],
    dx: npt.NDArray[float],
    dx_off: tuple[float, float],
    dy: npt.NDArray[float],
    dy_off: tuple[float, float],
    h: npt.NDArray[float],
    h_off: tuple[float, float],
    C: npt.NDArray[float],
    C_off: tuple[float],
    zeta: npt.NDArray[float],
    zeta_off: tuple[float, float],
) -> None:
    """First rk2 step."""

    # compute z at current time 
    h_idx = offset_indices_2D(particle["yidx"], particle["xidx"], h_off)
    C_idx = offset_indices_1D(particle["zidx"], C_off)
    zeta_idx = offset_indices_2D(particle["yidx"], particle["xidx"], zeta_off)   
    h_value = bilinear_interpolation(h_idx, h)
    C_value = linear_interpolation(C_idx, C)
    zeta_value = bilinear_interpolation(zeta_idx, zeta)

    sigma = sigma_coordinate(particle["zidx"], NZ)
    S = S_coordinate(hc, sigma, h_value, C_value)
    particle["z"] = z_from_S(S, h_value, zeta_value)

    # horizontal advection in index space
    # offset indices for u, v, dx, dy
    u_idx = offset_indices_3D(particle["zidx"], particle["yidx"], particle["xidx"], u_off)
    v_idx = offset_indices_3D(particle["zidx"], particle["yidx"], particle["xidx"], v_off)
    dx_idx = offset_indices_2D(particle["yidx"], particle["xidx"], dx_off)
    dy_idx = offset_indices_2D(particle["yidx"], particle["xidx"], dy_off)

    # interpolate u, v, dx, dy onto particle
    u_interp = trilinear_interpolation(u_idx, u)
    v_interp = trilinear_interpolation(v_idx, v)
    dx_interp = bilinear_interpolation(dx_idx, dx)
    dy_interp = bilinear_interpolation(dy_idx, dy)

    # compute rate of change of indices
    particle["_dxidx1"] = u_interp / dx_interp
    particle["_dyidx1"] = v_interp / dy_interp

    # vertical advection in physical space
    w_idx = offset_indices_3D(particle["zidx"], particle["yidx"], particle["xidx"], w_off)
    w_interp = trilinear_interpolation(w_idx, w)
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
    scalars=("hc", "NZ"),
    simulation_fields=[
        "u",
        "v",
        "w",
        "dx",
        "dy",
        "h",
        "C",
        "zeta",
    ],
)


def _rk2_step_2(
    particle: npt.NDArray,
    dt: float,
    alpha: float,
    hc: float,
    NZ: int,
    u: npt.NDArray[float], 
    u_off: tuple[float, float, float],
    v: npt.NDArray[float],
    v_off: tuple[float, float, float],
    w: npt.NDArray[float],
    w_off: tuple[float, float, float],
    dx: npt.NDArray[float],
    dx_off: tuple[float, float],
    dy: npt.NDArray[float],
    dy_off: tuple[float, float],
    h: npt.NDArray[float],
    h_off: tuple[float, float],
    C: npt.NDArray[float],
    C_off: tuple[float],
    zeta: npt.NDArray[float],
    zeta_off: tuple[float, float],
) -> None:
    """Second rk2 step."""

    # intermediate z position 
    z = particle["z"] + particle["_dz1"] * dt * alpha

    # intermediate xidx, yidx positions
    xidx = particle["xidx"] + particle["_dxidx1"] * dt * alpha
    yidx = particle["yidx"] + particle["_dyidx1"] * dt * alpha

    # intermediate zidx value 
    h_idx = offset_indices_2D(yidx, xidx, h_off)
    zeta_idx = offset_indices_2D(yidx, xidx, zeta_off)
    h_value = bilinear_interpolation(h_idx, h)
    zeta_value = bilinear_interpolation(zeta_idx, zeta)
    
    S = S_from_z(z, h_value, zeta_value)
    zidx = compute_zidx_from_S(S, hc, NZ, h_value, zeta_value, C, C_off)
    
    # horizontal advection in index space
    # offset indices for u, v, dx, dy
    u_idx = offset_indices_3D(zidx, yidx, xidx, u_off)
    v_idx = offset_indices_3D(zidx, yidx, xidx, v_off)
    dx_idx = offset_indices_2D(yidx, xidx, dx_off)
    dy_idx = offset_indices_2D(yidx, xidx, dy_off)

    # interpolate u, v, dx, dy onto particle
    u_interp = trilinear_interpolation(u_idx, u)
    v_interp = trilinear_interpolation(v_idx, v)
    dx_interp = bilinear_interpolation(dx_idx, dx)
    dy_interp = bilinear_interpolation(dy_idx, dy)

    # compute rate of change of indices
    particle["_dxidx2"] = u_interp / dx_interp
    particle["_dyidx2"] = v_interp / dy_interp

    # vertical advection in physical space
    w_idx = offset_indices_3D(zidx, yidx, xidx, w_off)
    w_interp = trilinear_interpolation(w_idx, w)
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
    scalars=("_dt", "_RK2_alpha", "hc", "NZ"),
    simulation_fields=[
        "u",
        "v",
        "w",
        "dx",
        "dy",
        "h",
        "C",
        "zeta",
    ],
)

def _rk2_update(
    particle: npt.NDArray,
    dt: float,
    alpha: float,
    hc: float,
    NZ: float,
    h: npt.NDArray[float],
    h_off: tuple[float, float],
    C: npt.NDArray[float],
    C_off: tuple[float],
    zeta: npt.NDArray[float],
    zeta_off: tuple[float, float],
) -> None:
    """RK2 update step."""
    b1 = 1.0 - 1.0 / (2.0 * alpha)
    b2 = 1.0 / (2.0 * alpha)

    # update horizontal indices
    particle["xidx"] += b1 * dt * particle["_dxidx1"] + b2 * dt * particle["_dxidx2"]
    particle["yidx"] += b1 * dt * particle["_dyidx1"] + b2 * dt * particle["_dyidx2"]

    # update vertical position
    particle["z"] = particle["z"] + b1 * dt * particle["_dz1"] + b2 * dt * particle["_dz2"]

    # compute new zidx from updated z
    h_idx = offset_indices_2D(particle["yidx"], particle["xidx"], h_off)
    zeta_idx = offset_indices_2D(particle["yidx"], particle["xidx"], zeta_off)
    h_value = bilinear_interpolation(h_idx, h)
    zeta_value = bilinear_interpolation(zeta_idx, zeta)
    
    S = S_from_z(particle["z"], h_value, zeta_value)
    particle["zidx"] = compute_zidx_from_S(S, hc, NZ, h_value, zeta_value, C, C_off)

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
    scalars=("_dt", "_RK2_alpha", "hc", "NZ"),
    simulation_fields=[
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
