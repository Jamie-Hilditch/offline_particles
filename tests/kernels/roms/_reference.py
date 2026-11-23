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

_C_OFFSET = 0.5

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


# --- sigma coordinate at w points, mirroring _vertical_coordinate.py ---


# mirrors _sigma_coordinate_linear
def sigma_coordinate_linear(zidx: float, Nz: int) -> float:
    return (zidx + 0.5) / Nz - 1.0


# mirrors _sigma_coordinate_quadratic
def sigma_coordinate_quadratic(zidx: float, Nz: int) -> float:
    x = zidx + 0.5
    N = float(Nz)
    return -(x**2 - 2.0 * x * N + x + N**2 - N) / (N**2 - N) - 0.01 * (x**2 - x * N) / (1.0 - N)


# mirrors _sigma_w_array
def sigma_w_array(NZ: int, sigma_method: str) -> npt.NDArray[np.float64]:
    sigma_fn = sigma_coordinate_quadratic if sigma_method == "quadratic" else sigma_coordinate_linear
    k = np.arange(NZ + 1, dtype=np.float64)
    return np.array([sigma_fn(ki - _C_OFFSET, NZ) for ki in k])


# --- core S-coordinate transform, mirroring _vertical_coordinate.py ---


# mirrors _S_coordinate
def S_coordinate(hc: float, sigma: float, h: float, C: float) -> float:
    return (hc * sigma + h * C) / (hc + h)


# mirrors _z_coordinate
def z_coordinate(S: float, h: float, zeta: float) -> float:
    return zeta + (zeta + h) * S


# mirrors compute_z: physical vertical coordinate from already-resolved sigma and C values
def compute_z(sigma: float, hc: float, h: float, C: float, zeta: float) -> float:
    S = S_coordinate(hc, sigma, h, C)
    return z_coordinate(S, h, zeta)


# mirrors _S_from_z
def S_from_z(z: float, h: float, zeta: float) -> float:
    return (z - zeta) / (zeta + h)


# mirrors _compute_Cidx_from_S: binary search over the stretching function array
def compute_Cidx_from_S(
    S: float,
    hc: float,
    h: float,
    sigma_w: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
) -> int:
    C_size = C.shape[0]
    lo = 0
    hi = C_size - 2

    # boundary clamps, exactly as documented in the kernel implementation's docstring
    if S <= S_coordinate(hc, sigma_w[0], h, C[0]):
        return 0
    if S >= S_coordinate(hc, sigma_w[C_size - 2], h, C[C_size - 2]):
        return C_size - 2

    while lo < hi - 1:
        mid = (lo + hi) // 2
        S_mid = S_coordinate(hc, sigma_w[mid], h, C[mid])
        if S_mid <= S:
            lo = mid
        else:
            hi = mid
    return lo


# mirrors _zidx_from_S
def zidx_from_S(
    S: float,
    hc: float,
    h: float,
    sigma_w: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
) -> float:
    Cidx = compute_Cidx_from_S(S, hc, h, sigma_w, C)
    S_low = S_coordinate(hc, sigma_w[Cidx], h, C[Cidx])
    S_high = S_coordinate(hc, sigma_w[Cidx + 1], h, C[Cidx + 1])
    f = (S - S_low) / (S_high - S_low)
    return Cidx - _C_OFFSET + f


# mirrors compute_zidx: vertical index from the physical vertical coordinate
def compute_zidx(
    z: float,
    h: float,
    zeta: float,
    hc: float,
    sigma_w: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
) -> float:
    S = S_from_z(z, h, zeta)
    return zidx_from_S(S, hc, h, sigma_w, C)
