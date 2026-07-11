"""Pure-Python/NumPy reference oracle for the ROMS vertical-coordinate transform.

Mirrors, line for line, the equations implemented by the `numba`-jitted vertical-coordinate
kernel in ``src/offline_particles/kernels/roms/_vertical_coordinate.py``.
This gives that kernel an independent second implementation to be checked against, without
needing to reach into its private helper functions.

If the vertical-coordinate kernels are ever rewritten, this module should keep mirroring
whatever equations that rewrite implements, so it continues to serve as ground truth.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from offline_particles.kernels.status import INACTIVE_FLAG

# --- interpolation helpers, mirroring the linear-interpolation kernel implementation ---


# mirrors truncate_index: floor (toward zero) and clamp to [0, max_idx]
def truncate_index(idx: float, max_idx: int) -> int:
    i = int(idx)
    if i < 0:
        i = 0
    elif i > max_idx:
        i = max_idx
    return i


# mirrors linear_interpolation
def linear_interpolation(array: npt.NDArray[np.float64], idx: float) -> np.float64:
    I0 = truncate_index(idx, array.shape[0] - 2)
    f0 = idx - I0
    g0 = 1.0 - f0
    return g0 * array[I0] + f0 * array[I0 + 1]


# mirrors bilinear_interpolation
def bilinear_interpolation(array: npt.NDArray[np.float64], idx0: float, idx1: float) -> np.float64:
    I0 = truncate_index(idx0, array.shape[0] - 2)
    I1 = truncate_index(idx1, array.shape[1] - 2)
    f0 = idx0 - I0
    f1 = idx1 - I1
    g0 = 1.0 - f0
    g1 = 1.0 - f1
    v00 = array[I0, I1]
    v01 = array[I0, I1 + 1]
    v10 = array[I0 + 1, I1]
    v11 = array[I0 + 1, I1 + 1]
    return g0 * g1 * v00 + g0 * f1 * v01 + f0 * g1 * v10 + f0 * f1 * v11


# --- core S-coordinate transform, mirroring _vertical_coordinate.py ---


# mirrors _sigma_coordinate
def sigma_coordinate(zidx: float, Nz: int) -> float:
    return (zidx + 0.5) / Nz - 1.0


# mirrors _S_coordinate
def S_coordinate(hc: float, sigma: float, h: float, C: float) -> float:
    return (hc * sigma + h * C) / (hc + h)


# mirrors _z_coordinate
def z_coordinate(S: float, h: float, zeta: float) -> float:
    return zeta + (zeta + h) * S


# mirrors compute_z: physical vertical coordinate from the vertical index
def compute_z(zidx: float, Nz: int, hc: float, h: float, C: float, zeta: float) -> float:
    sigma = sigma_coordinate(zidx, Nz)
    S = S_coordinate(hc, sigma, h, C)
    return z_coordinate(S, h, zeta)


# mirrors _S_from_z
def S_from_z(z: float, h: float, zeta: float) -> float:
    return (z - zeta) / (zeta + h)


# mirrors _sigma_from_Cidx
def sigma_from_Cidx(Cidx: int, C_offset: float, Nz: int) -> float:
    return sigma_coordinate(Cidx - C_offset, Nz)


# mirrors _compute_Cidx_from_S: binary search over the stretching function array
def compute_Cidx_from_S(
    S: float,
    hc: float,
    NZ: int,
    h: float,
    zeta: float,
    C: npt.NDArray[np.float64],
    C_offset: float,
) -> int:
    C_size = C.shape[0]
    lo = 0
    hi = C_size - 2

    # boundary clamps, exactly as documented in the kernel implementation's docstring
    if S <= S_coordinate(hc, sigma_from_Cidx(0, C_offset, NZ), h, C[0]):
        return 0
    if S >= S_coordinate(hc, sigma_from_Cidx(C_size - 2, C_offset, NZ), h, C[C_size - 2]):
        return C_size - 2

    while lo < hi - 1:
        mid = (lo + hi) // 2
        S_mid = S_coordinate(hc, sigma_from_Cidx(mid, C_offset, NZ), h, C[mid])
        if S_mid <= S:
            lo = mid
        else:
            hi = mid
    return lo


# mirrors _zidx_from_S
def zidx_from_S(
    S: float,
    hc: float,
    NZ: int,
    h: float,
    zeta: float,
    C: npt.NDArray[np.float64],
    C_offset: float,
) -> float:
    Cidx = compute_Cidx_from_S(S, hc, NZ, h, zeta, C, C_offset)
    S_low = S_coordinate(hc, sigma_from_Cidx(Cidx, C_offset, NZ), h, C[Cidx])
    S_high = S_coordinate(hc, sigma_from_Cidx(Cidx + 1, C_offset, NZ), h, C[Cidx + 1])
    f = (S - S_low) / (S_high - S_low)
    return Cidx - C_offset + f


# mirrors compute_zidx: vertical index from the physical vertical coordinate
def compute_zidx(
    z: float,
    h: float,
    zeta: float,
    hc: float,
    NZ: int,
    C: npt.NDArray[np.float64],
    C_offset: float,
) -> float:
    S = S_from_z(z, h, zeta)
    return zidx_from_S(S, hc, NZ, h, zeta, C, C_offset)


# --- whole-kernel reference, mirroring _vertical_coordinate.py ---


# reference re-implementation of compute_z_kernel_function
def reference_compute_z_kernel_function(particle_properties, hc, NZ, field_data) -> None:
    status = particle_properties["status"]
    zidx = particle_properties["zidx"]
    yidx = particle_properties["yidx"]
    xidx = particle_properties["xidx"]
    z = particle_properties["z"]

    h_array, h_offy, h_offx = field_data["h"].unpack()
    zeta_array, zeta_offy, zeta_offx = field_data["zeta"].unpack()
    C_array, C_offz = field_data["C"].unpack()

    for i in range(status.shape[0]):
        if status[i] & INACTIVE_FLAG:
            continue
        h_value = bilinear_interpolation(h_array, yidx[i] + h_offy, xidx[i] + h_offx)
        zeta_value = bilinear_interpolation(zeta_array, yidx[i] + zeta_offy, xidx[i] + zeta_offx)
        C_value = linear_interpolation(C_array, zidx[i] + C_offz)
        z[i] = compute_z(zidx[i], NZ, hc, h_value, C_value, zeta_value)


# reference re-implementation of compute_zidx_kernel_function
def reference_compute_zidx_kernel_function(particle_properties, hc, NZ, field_data) -> None:
    status = particle_properties["status"]
    zidx = particle_properties["zidx"]
    yidx = particle_properties["yidx"]
    xidx = particle_properties["xidx"]
    z = particle_properties["z"]

    h_array, h_offy, h_offx = field_data["h"].unpack()
    zeta_array, zeta_offy, zeta_offx = field_data["zeta"].unpack()
    C_array, C_offz = field_data["C"].unpack()

    for i in range(status.shape[0]):
        if status[i] & INACTIVE_FLAG:
            continue
        h_value = bilinear_interpolation(h_array, yidx[i] + h_offy, xidx[i] + h_offx)
        zeta_value = bilinear_interpolation(zeta_array, yidx[i] + zeta_offy, xidx[i] + zeta_offx)
        zidx[i] = compute_zidx(z[i], h_value, zeta_value, hc, NZ, C_array, C_offz)
