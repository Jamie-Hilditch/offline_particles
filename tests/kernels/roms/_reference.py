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

_W_OFFSET = 0.5

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
    return np.array([sigma_fn(ki - _W_OFFSET, NZ) for ki in k])


# --- ROMS vertical transform (tau), mirroring _vertical_coordinate.py ---


# mirrors _tau_Vtransform_1
def tau_Vtransform_1(hc: float, h: float) -> float:
    return hc / h


# mirrors _tau_Vtransform_2
def tau_Vtransform_2(hc: float, h: float) -> float:
    return hc / (hc + h)


# --- core S-coordinate transform, mirroring _vertical_coordinate.py ---


# mirrors _S_coordinate
def S_coordinate(tau: float, sigma: float, C: float) -> float:
    return tau * sigma + (1.0 - tau) * C


# mirrors _z_coordinate
def z_coordinate(S: float, h: float, zeta: float) -> float:
    return zeta + (zeta + h) * S


# mirrors compute_z: physical vertical coordinate from already-resolved sigma and C values
def compute_z(sigma: float, tau: float, h: float, C: float, zeta: float) -> float:
    S = S_coordinate(tau, sigma, C)
    return z_coordinate(S, h, zeta)


# mirrors _S_from_z
def S_from_z(z: float, h: float, zeta: float) -> float:
    return (z - zeta) / (zeta + h)


# mirrors _compute_widx_from_S: binary search over the stretching function array
def compute_widx_from_S(
    S: float,
    tau: float,
    sigma_w: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
) -> int:
    C_size = C.shape[0]
    lo = 0
    hi = C_size - 2

    # boundary clamps, exactly as documented in the kernel implementation's docstring
    if S <= S_coordinate(tau, sigma_w[0], C[0]):
        return 0
    if S >= S_coordinate(tau, sigma_w[C_size - 2], C[C_size - 2]):
        return C_size - 2

    while lo < hi - 1:
        mid = (lo + hi) // 2
        S_mid = S_coordinate(tau, sigma_w[mid], C[mid])
        if S_mid <= S:
            lo = mid
        else:
            hi = mid
    return lo


# mirrors _linear_zidx_from_S
def linear_zidx_from_S(
    S: float,
    tau: float,
    sigma_w: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
) -> float:
    widx = compute_widx_from_S(S, tau, sigma_w, C)
    S_low = S_coordinate(tau, sigma_w[widx], C[widx])
    S_high = S_coordinate(tau, sigma_w[widx + 1], C[widx + 1])
    f = (S - S_low) / (S_high - S_low)
    return widx - _W_OFFSET + f


# mirrors compute_zidx_linear: vertical index from the physical vertical coordinate, linearly
def compute_zidx_linear(
    z: float,
    h: float,
    zeta: float,
    tau: float,
    sigma_w: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
) -> float:
    S = S_from_z(z, h, zeta)
    return linear_zidx_from_S(S, tau, sigma_w, C)


# --- monotone cubic Hermite interpolation, mirroring _vertical_coordinate.py ---


# mirrors _hermite_cubic_coefficients
def hermite_cubic_coefficients(f0: float, f1: float, d0: float, d1: float) -> tuple[float, float, float, float]:
    a = 2.0 * f0 + d0 - 2.0 * f1 + d1
    b = -3.0 * f0 - 2.0 * d0 + 3.0 * f1 - d1
    return a, b, d0, f0


# mirrors _cubic_eval
def cubic_eval(a: float, b: float, c: float, d: float, t: float) -> float:
    return ((a * t + b) * t + c) * t + d


# mirrors _cubic_eval_with_derivatives
def cubic_eval_with_derivatives(a: float, b: float, c: float, d: float, t: float) -> tuple[float, float, float]:
    value = ((a * t + b) * t + c) * t + d
    first_derivative = (3.0 * a * t + 2.0 * b) * t + c
    second_derivative = 6.0 * a * t + 2.0 * b
    return value, first_derivative, second_derivative


# mirrors _hermite_interpolation_particle
def hermite_interpolation(
    value_array: npt.NDArray[np.float64],
    derivative_array: npt.NDArray[np.float64],
    offset_idx: float,
) -> float:
    I0 = truncate_index(offset_idx, value_array.shape[0] - 2)
    t = offset_idx - I0
    a, b, c, d = hermite_cubic_coefficients(
        value_array[I0], value_array[I0 + 1], derivative_array[I0], derivative_array[I0 + 1]
    )
    return cubic_eval(a, b, c, d, t)


# mirrors _bessel_node_derivative_array
def bessel_node_derivative_array(f: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    delta = np.diff(f)
    derivative = np.empty_like(f)
    derivative[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    derivative[0] = (3.0 * delta[0] - delta[1]) / 2.0
    derivative[-1] = (3.0 * delta[-1] - delta[-2]) / 2.0
    return derivative


# mirrors _hyman_filter_node_derivative_array
def hyman_filter_node_derivative_array(
    f: npt.NDArray[np.float64], raw_derivative: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    delta = np.diff(f)
    derivative = np.empty_like(f)
    derivative[1:-1] = np.minimum(raw_derivative[1:-1], 3.0 * np.minimum(delta[:-1], delta[1:]))
    derivative[0] = min(raw_derivative[0], 3.0 * delta[0]) if np.sign(raw_derivative[0]) == np.sign(delta[0]) else 0.0
    derivative[-1] = (
        min(raw_derivative[-1], 3.0 * delta[-1]) if np.sign(raw_derivative[-1]) == np.sign(delta[-1]) else 0.0
    )
    return derivative


# mirrors _node_derivative_array
def node_derivative_array(f: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return hyman_filter_node_derivative_array(f, bessel_node_derivative_array(f))


# mirrors _cubic_zidx_from_S_factory's inner function
def cubic_zidx_from_S(
    S: float,
    tau: float,
    sigma_w: npt.NDArray[np.float64],
    sigma_w_derivative: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
    C_derivative: npt.NDArray[np.float64],
    halley_iterations: int,
) -> float:
    widx = compute_widx_from_S(S, tau, sigma_w, C)

    S_low = S_coordinate(tau, sigma_w[widx], C[widx])
    S_high = S_coordinate(tau, sigma_w[widx + 1], C[widx + 1])
    t = (S - S_low) / (S_high - S_low)

    a_sigma, b_sigma, c_sigma, d_sigma = hermite_cubic_coefficients(
        sigma_w[widx], sigma_w[widx + 1], sigma_w_derivative[widx], sigma_w_derivative[widx + 1]
    )
    a_C, b_C, c_C, d_C = hermite_cubic_coefficients(C[widx], C[widx + 1], C_derivative[widx], C_derivative[widx + 1])
    a = tau * a_sigma + (1.0 - tau) * a_C
    b = tau * b_sigma + (1.0 - tau) * b_C
    c = tau * c_sigma + (1.0 - tau) * c_C
    d = tau * d_sigma + (1.0 - tau) * d_C - S

    for _ in range(halley_iterations):
        g, g_prime, g_double_prime = cubic_eval_with_derivatives(a, b, c, d, t)
        t = t - 2.0 * g * g_prime / (2.0 * g_prime * g_prime - g * g_double_prime)

    return widx - _W_OFFSET + t


# mirrors _compute_zidx_cubic_factory's inner function: vertical index from the physical vertical
# coordinate, via cubic Hermite interpolation
def compute_zidx_cubic(
    z: float,
    h: float,
    zeta: float,
    tau: float,
    sigma_w: npt.NDArray[np.float64],
    sigma_w_derivative: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
    C_derivative: npt.NDArray[np.float64],
    halley_iterations: int,
) -> float:
    S = S_from_z(z, h, zeta)
    return cubic_zidx_from_S(S, tau, sigma_w, sigma_w_derivative, C, C_derivative, halley_iterations)
