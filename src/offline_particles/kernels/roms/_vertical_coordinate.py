"""Numba functions for working with ROMS' sigma coordinate scheme.

We follow the same variable naming conventions as ROMS:
- zidx: vertical index
- Nz: total number of vertical rho levels
- hc: critical depth
- sigma: sigma coordinate (between -1 and 0)
- h: bathymetric depth
- C: stretching function value
- S: S-coordinate value (stretched sigma coordinate)
- zeta: free surface elevation
- z: physical vertical coordinate

Both sigma and C (the stretching function) are defined only at the `NZ + 1` w points. `S`, and
therefore `z`, is defined on the continuous vertical index by interpolating between those w-point
values - linearly, for `LinearVerticalTransform`, or via a monotone cubic Hermite spline, for
`CubicVerticalTransform`. `compute_zidx_linear` is an *exact* inverse of `compute_z` on every grid
cell, for either `sigma_method`, because both directions operate on the same interpolated
representation of `S`; `compute_zidx_cubic` inverts the same `S` to within a small, fixed-iteration
Halley residual (see `CubicVerticalTransform`'s docstring).

Notes
-----
ROMS supports two vertical transforms (`Vtransform`). Both reduce to the same shape once `S` is
defined as a nondimensional quantity: `z = zeta + (zeta + h) * S`, where
`S = tau * sigma + (1 - tau) * C` and `tau` depends only on `hc` and `h`:
`tau = hc / h` for `Vtransform=1`, `tau = hc / (hc + h)` for `Vtransform=2`. ROMS's own
`Vtransform=2` formula, `(hc*sigma + h*C) / (hc + h)`, is already this nondimensional `S` exactly.
ROMS's own `Vtransform=1` formula instead produces a dimensional quantity (call it `Zo`, units of
length): `Zo = hc*sigma + (h-hc)*C`, `z = Zo + zeta*(1 + Zo/h)`; dividing through by `h` gives
`S := Zo/h`, which is what makes the two transforms share one `z`/`S` formula here.

`C` (the vertical stretching function) must be strictly increasing along the vertical axis. This
is a physical requirement of the ROMS S-coordinate system: a valid vertical grid never has
degenerate or crossing levels. `compute_zidx_linear`'s binary search over `C` assumes this;
behavior is undefined if it does not hold. The same requirement applies to the tabulated `sigma_w`
array built internally from `sigma_method` - guaranteed by construction for both the linear and
quadratic formulae, for any `NZ >= 2`, and not re-validated at runtime.
"""

import abc
from dataclasses import dataclass, field, fields
from typing import Literal, Self

import numba
import numpy as np
import numpy.typing as npt

from .._kernels import KernelFunction, kernel_function
from ..interpolation import lagrange_2D_particle_factory, linear_interpolation_particle
from ..status import INACTIVE_FLAG, Status

_INITIALISING = np.uint8(Status.INITIALISING)

# the w-point-to-rho-point-index offset, shared by sigma_w and C alike: w-point j's sigma/C value
# equals rho-point (j - 0.5)'s
_W_OFFSET = 0.5

#####################################
### sigma coordinate at w points  ###
#####################################


@numba.njit(nogil=True, fastmath=True)
def _sigma_coordinate_linear(zidx: float, Nz: int) -> float:
    """Linear sigma coordinate (ROMS Vstretching 1-4): sc_w(k) = (k - N) / N at zidx = k - 0.5.

    Parameters
    ----------
    zidx : float
        The vertical index.
    Nz : int
        The total number of vertical rho levels.

    Returns
    -------
    float
        The sigma coordinate.
    """
    return (zidx + 0.5) / Nz - 1.0


@numba.njit(nogil=True, fastmath=True)
def _sigma_coordinate_quadratic(zidx: float, Nz: int) -> float:
    r"""Quadratic sigma coordinate (ROMS Vstretching=5, Souza et al. 2015).

    Continuous extension (via `zidx = k - 0.5`) of

    .. math::

        sc_w(k) = -\frac{k^2 - 2kN + k + N^2 - N}{N^2 - N} - 0.01 \frac{k^2 - kN}{1 - N}

    for `k = 0, ..., N`. Satisfies `sc_w(0) = -1`, `sc_w(N) = 0`, with quadratic (rather than
    linear) spacing in between that clusters more levels near the surface.

    Parameters
    ----------
    zidx : float
        The vertical index.
    Nz : int
        The total number of vertical rho levels.

    Returns
    -------
    float
        The sigma coordinate.

    References
    ----------
    Souza, J., B.S. Powell, A.C. Castillo-Trujillo, and P. Flament, 2015: The Vorticity Balance
    of the Ocean Surface in Hawaii from a Regional Reanalysis. J. Phys. Oceanogr., 45, 424-440.
    """
    x = zidx + 0.5
    N = float(Nz)
    return -(x**2 - 2.0 * x * N + x + N**2 - N) / (N**2 - N) - 0.01 * (x**2 - x * N) / (1.0 - N)


def _sigma_w_array(NZ: int, sigma_method: str) -> npt.NDArray[np.float64]:
    """Tabulate the sigma coordinate at the `NZ + 1` w points.

    Dispatches once (not per particle) on `sigma_method` between the linear and quadratic sigma
    formulae.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels.
    sigma_method : str
        Either "linear" or "quadratic".

    Returns
    -------
    npt.NDArray[np.float64]
        The sigma coordinate at each of the `NZ + 1` w points.
    """
    sigma_fn = _sigma_coordinate_quadratic if sigma_method == "quadratic" else _sigma_coordinate_linear
    k = np.arange(NZ + 1, dtype=np.float64)
    return np.array([sigma_fn(ki - _W_OFFSET, NZ) for ki in k])


#################################################
### ROMS vertical transform (tau)            ###
#################################################


@numba.njit(nogil=True, fastmath=True)
def _tau_Vtransform_1(hc: float, h: float) -> float:
    """tau(hc, h) for ROMS Vtransform=1 (Song & Haidvogel 1994 lineage): hc / h.

    Parameters
    ----------
    hc : float
        The critical depth.
    h : float
        The bathymetric depth.

    Returns
    -------
    float
        The blend weight `tau` used by `_S_coordinate`.
    """
    return hc / h


@numba.njit(nogil=True, fastmath=True)
def _tau_Vtransform_2(hc: float, h: float) -> float:
    """tau(hc, h) for ROMS Vtransform=2 (Shchepetkin & McWilliams 2005): hc / (hc + h).

    Parameters
    ----------
    hc : float
        The critical depth.
    h : float
        The bathymetric depth.

    Returns
    -------
    float
        The blend weight `tau` used by `_S_coordinate`.
    """
    return hc / (hc + h)


def _tau_fn_for_Vtransform(Vtransform: int):
    """Select the tau(hc, h) numba dispatcher for the given ROMS Vtransform.

    Parameters
    ----------
    Vtransform : int
        Either 1 or 2.

    Returns
    -------
    Callable[[float, float], float]
        `_tau_Vtransform_1` or `_tau_Vtransform_2`.

    Raises
    ------
    ValueError
        If `Vtransform` is not 1 or 2.
    """
    if Vtransform == 1:
        return _tau_Vtransform_1
    elif Vtransform == 2:
        return _tau_Vtransform_2
    raise ValueError(f"Invalid Vtransform: {Vtransform!r}. Must be 1 or 2.")


#############################
### computing z from zidx ###
#############################


@numba.njit(nogil=True, fastmath=True)
def _S_coordinate(tau: float, sigma: float, C: float) -> float:
    """Compute the S-coordinate transformation for ROMS vertical coordinates.

    Parameters
    ----------
    tau : float
        The blend weight, depending on the ROMS vertical transform (`tau = hc/h` for
        Vtransform=1, `tau = hc/(hc+h)` for Vtransform=2).
    sigma : float
        The sigma coordinate.
    C : float
        The stretching function value.

    Returns
    -------
    float
        The S-coordinate value.
    """
    return tau * sigma + (1.0 - tau) * C


@numba.njit(nogil=True, fastmath=True)
def _z_coordinate(S: float, h: float, zeta: float) -> float:
    """Convert S-coordinate to physical coordinate.

    Parameters
    ----------
    S : float
        The S-coordinate value.
    h : float
        The bathymetric depth.
    zeta : float
        The free surface elevation.

    Returns
    -------
    float
        The physical vertical coordinate.
    """
    return zeta + (zeta + h) * S


@numba.njit(nogil=True, fastmath=True)
def compute_z(sigma: float, tau: float, h: float, C: float, zeta: float) -> float:
    """Compute the physical vertical coordinate from already-resolved sigma and C values.

    `sigma` and `C` must already be resolved at the particle's vertical index (e.g. by linearly or
    cubic-Hermite interpolating the tabulated `sigma_w`/`Cs_w` arrays) - this function only applies
    the S-coordinate and z-coordinate transforms.

    Parameters
    ----------
    sigma : float
        The sigma coordinate value.
    tau : float
        The blend weight, depending on the ROMS vertical transform (`tau = hc/h` for
        Vtransform=1, `tau = hc/(hc+h)` for Vtransform=2).
    h : float
        The bathymetric depth.
    C : float
        The stretching function value.
    zeta : float
        The free surface elevation.

    Returns
    -------
    float
        The physical vertical coordinate.
    """
    S = _S_coordinate(tau, sigma, C)
    return _z_coordinate(S, h, zeta)


#############################
### computing zidx from z ###
#############################


@numba.njit(nogil=True, fastmath=True)
def _S_from_z(z: float, h: float, zeta: float) -> float:
    """Convert physical coordinate to S-coordinate.

    Parameters
    ----------
    z : float
        The physical vertical coordinate.
    h : float
        The bathymetric depth.
    zeta : float
        The free surface elevation.

    Returns
    -------
    float
        The S-coordinate value.
    """
    return (z - zeta) / (zeta + h)


@numba.njit(nogil=True, fastmath=True)
def _compute_widx_from_S(
    S: float,
    tau: float,
    sigma_w: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
) -> int:
    """Binary search for the w-point index bracketing the given S coordinate.

    Returns `widx` such that `_S_coordinate(tau, sigma_w[widx], C[widx]) <= S` and
    `_S_coordinate(tau, sigma_w[widx + 1], C[widx + 1]) > S`. This assumes `C` is strictly
    increasing (see the module docstring). If `S` is outside the range of S values defined by
    `C`, the first or penultimate index is returned.

    Parameters
    ----------
    S : float
        The S-coordinate value.
    tau : float
        The blend weight, depending on the ROMS vertical transform (`tau = hc/h` for
        Vtransform=1, `tau = hc/(hc+h)` for Vtransform=2).
    sigma_w : npt.NDArray[np.float64]
        The sigma coordinate tabulated at the w points. Must have the same length as `C`.
    C : npt.NDArray[np.float64]
        The stretching function array. Must be strictly increasing.

    Returns
    -------
    int
        The index into `C`/`sigma_w` bracketing S, clamped to [0, C.shape[0] - 2].
    """
    C_size = C.shape[0]
    lo = 0
    hi = C_size - 2

    # handle edge cases where S is outside the range of S values
    if S <= _S_coordinate(tau, sigma_w[0], C[0]):
        return 0
    elif S >= _S_coordinate(tau, sigma_w[C_size - 2], C[C_size - 2]):
        return C_size - 2

    # binary search
    while lo < hi - 1:
        mid = (lo + hi) // 2
        S_mid = _S_coordinate(tau, sigma_w[mid], C[mid])
        if S_mid <= S:
            lo = mid
        else:
            hi = mid
    return lo


@numba.njit(nogil=True, fastmath=True)
def _linear_zidx_from_S(
    S: float,
    tau: float,
    sigma_w: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
) -> float:
    """Fractional vertical index, linearly interpolating S within the bracketing w-point cell.

    Parameters
    ----------
    S : float
        The S-coordinate value.
    tau : float
        The blend weight, depending on the ROMS vertical transform (`tau = hc/h` for
        Vtransform=1, `tau = hc/(hc+h)` for Vtransform=2).
    sigma_w : npt.NDArray[np.float64]
        The sigma coordinate tabulated at the w points. Must have the same length as `C`.
    C : npt.NDArray[np.float64]
        The stretching function array. Must be strictly increasing.

    Returns
    -------
    float
        The vertical index.
    """
    widx = _compute_widx_from_S(S, tau, sigma_w, C)
    S_low = _S_coordinate(tau, sigma_w[widx], C[widx])
    S_high = _S_coordinate(tau, sigma_w[widx + 1], C[widx + 1])
    f = (S - S_low) / (S_high - S_low)
    return widx - _W_OFFSET + f


@numba.njit(nogil=True, fastmath=True)
def compute_zidx_linear(
    z: float,
    h: float,
    zeta: float,
    tau: float,
    sigma_w: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
) -> float:
    """Compute the vertical index from the physical vertical coordinate, via linear interpolation.

    Parameters
    ----------
    z : float
        The physical vertical coordinate.
    h : float
        The bathymetric depth.
    zeta : float
        The free surface elevation.
    tau : float
        The blend weight, depending on the ROMS vertical transform (`tau = hc/h` for
        Vtransform=1, `tau = hc/(hc+h)` for Vtransform=2).
    sigma_w : npt.NDArray[np.float64]
        The sigma coordinate tabulated at the w points. Must have the same length as `C`.
    C : npt.NDArray[np.float64]
        The stretching function array. Must be strictly increasing (see the module docstring);
        behavior is undefined otherwise.

    Returns
    -------
    float
        The vertical index.
    """
    S = _S_from_z(z, h, zeta)
    return _linear_zidx_from_S(S, tau, sigma_w, C)


#################################################
### ROMS stretching functions (Vstretching)  ###
#################################################


def _Cs_w_Vstretching_1(NZ: int, theta_s: float, theta_b: float) -> npt.NDArray[np.float64]:
    r"""Build `Cs_w` using ROMS Vstretching=1 (Song & Haidvogel 1994), with linear sigma.

    .. math::

        C_s(\sigma) = (1 - \theta_b) \frac{\sinh(\theta_s \sigma)}{\sinh(\theta_s)}
        + \theta_b \left[ -0.5 + 0.5 \frac{\tanh(\theta_s(\sigma + 0.5))}{\tanh(0.5\theta_s)} \right]

    with :math:`C_s(\sigma) = \sigma` if :math:`\theta_s = 0`. Valid for
    :math:`0 < \theta_s \le 8`, :math:`0 \le \theta_b \le 1`.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels.
    theta_s : float
        Surface control parameter.
    theta_b : float
        Bottom control parameter.

    Returns
    -------
    npt.NDArray[np.float64]
        `Cs_w`, tabulated at the `NZ + 1` w points.

    References
    ----------
    Song, Y. and D. Haidvogel, 1994: A semi-implicit ocean circulation model using a generalized
    topography-following coordinate system. J. Comp. Physics, 115, 228-244.
    """
    sigma_w = _sigma_w_array(NZ, "linear")
    if theta_s == 0:
        return sigma_w.copy()
    return (1 - theta_b) * np.sinh(theta_s * sigma_w) / np.sinh(theta_s) + theta_b * (
        -0.5 + 0.5 * np.tanh(theta_s * (sigma_w + 0.5)) / np.tanh(0.5 * theta_s)
    )


def _Cs_w_Vstretching_2(NZ: int, theta_s: float, theta_b: float) -> npt.NDArray[np.float64]:
    r"""Build `Cs_w` using ROMS Vstretching=2 (Shchepetkin & McWilliams 2005), with linear sigma.

    .. math::

        C_{sur}(\sigma) = \frac{1 - \cosh(\theta_s \sigma)}{\cosh(\theta_s) - 1}

    if :math:`\theta_s > 0`, else :math:`C_{sur}(\sigma) = \sigma`. If :math:`\theta_b > 0`,
    blended with a bottom-intensifying function:

    .. math::

        C_{bot}(\sigma) = \frac{\sinh(\theta_b(\sigma+1))}{\sinh(\theta_b)} - 1 \qquad
        C_{weight} = (\sigma+1)\left[2-(\sigma+1)\right] \qquad
        C_s(\sigma) = C_{weight} C_{sur}(\sigma) + (1 - C_{weight}) C_{bot}(\sigma)

    otherwise :math:`C_s(\sigma) = C_{sur}(\sigma)`. Unlike Vstretching=1, satisfies
    :math:`dC_s/d\sigma \to 0` as :math:`\sigma \to 0`.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels.
    theta_s : float
        Surface control parameter.
    theta_b : float
        Bottom control parameter.

    Returns
    -------
    npt.NDArray[np.float64]
        `Cs_w`, tabulated at the `NZ + 1` w points.

    References
    ----------
    Shchepetkin, A.F. and J.C. McWilliams, 2005: The regional oceanic modeling system (ROMS): a
    split-explicit, free-surface, topography-following-coordinate oceanic model. Ocean Modelling,
    9, 347-404.
    """
    sigma_w = _sigma_w_array(NZ, "linear")
    if theta_s > 0:
        Csur = (1 - np.cosh(theta_s * sigma_w)) / (np.cosh(theta_s) - 1)
    else:
        Csur = sigma_w.copy()
    if theta_b > 0:
        Cbot = np.sinh(theta_b * (sigma_w + 1)) / np.sinh(theta_b) - 1
        Cweight = (sigma_w + 1) * (2 - (sigma_w + 1))  # Aweight = Bweight = 1.0
        return Cweight * Csur + (1 - Cweight) * Cbot
    return Csur


def _Cs_w_Vstretching_3(NZ: int, theta_s: float, theta_b: float) -> npt.NDArray[np.float64]:
    r"""Build `Cs_w` using ROMS Vstretching=3 (R. Geyer), with linear sigma.

    Designed for high bottom boundary-layer resolution in shallow coastal applications.
    `theta_s`/`theta_b` are reinterpreted as surface/bottom stretching exponents
    :math:`\alpha`/:math:`\beta`:

    .. math::

        C_{sur}(\sigma) = -\frac{\ln\cosh\left(H|\sigma|^\alpha\right)}{\ln\cosh(H)} \qquad
        C_{bot}(\sigma) = \frac{\ln\cosh\left(H(\sigma+1)^\beta\right)}{\ln\cosh(H)} - 1 \qquad
        C_{weight} = 0.5\left[1 - \tanh\left(H(\sigma+0.5)\right)\right]

    .. math::

        C_s(\sigma) = C_{weight} C_{bot}(\sigma) + (1 - C_{weight}) C_{sur}(\sigma)

    with :math:`H = 3.0` fixed. :math:`\alpha`: 0.65 (minimal) to 1.0 (significant) surface
    stretching. :math:`\beta`: 0.58 (none) to 3.0 (super-high) bottom stretching.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels.
    theta_s : float
        Surface stretching exponent (alpha).
    theta_b : float
        Bottom stretching exponent (beta).

    Returns
    -------
    npt.NDArray[np.float64]
        `Cs_w`, tabulated at the `NZ + 1` w points.
    """
    sigma_w = _sigma_w_array(NZ, "linear")
    Hscale = 3.0
    alpha, beta = theta_s, theta_b
    Csur = -np.log(np.cosh(Hscale * np.abs(sigma_w) ** alpha)) / np.log(np.cosh(Hscale))
    Cbot = np.log(np.cosh(Hscale * (sigma_w + 1) ** beta)) / np.log(np.cosh(Hscale)) - 1
    Cweight = 0.5 * (1 - np.tanh(Hscale * (sigma_w + 0.5)))
    return Cweight * Cbot + (1 - Cweight) * Csur


def _C_double_stretching(sigma_w: npt.NDArray[np.float64], theta_s: float, theta_b: float) -> npt.NDArray[np.float64]:
    r"""ROMS Vstretching=4/5 "double stretching" C(s) formula, shared by both variants.

    .. math::

        C_{sur}(\sigma) = \frac{1 - \cosh(\theta_s \sigma)}{\cosh(\theta_s) - 1}

    if :math:`\theta_s > 0`, else :math:`C_{sur}(\sigma) = -\sigma^2`. If :math:`\theta_b > 0`:

    .. math::

        C_s(\sigma) = \frac{\exp(\theta_b C_{sur}(\sigma)) - 1}{1 - \exp(-\theta_b)}

    otherwise :math:`C_s(\sigma) = C_{sur}(\sigma)`. Valid for :math:`0 < \theta_s \le 10`,
    :math:`0 \le \theta_b \le 3`. Continuous in both parameters as either tends to zero.

    Parameters
    ----------
    sigma_w : npt.NDArray[np.float64]
        The sigma coordinate tabulated at the w points (linear for Vstretching=4, quadratic for
        Vstretching=5).
    theta_s : float
        Surface control parameter.
    theta_b : float
        Bottom control parameter.

    Returns
    -------
    npt.NDArray[np.float64]
        `Cs_w`, tabulated at the same w points as `sigma_w`.

    References
    ----------
    Shchepetkin, A.F. and J.C. McWilliams, 2005: The regional oceanic modeling system (ROMS): a
    split-explicit, free-surface, topography-following-coordinate oceanic model. Ocean Modelling,
    9, 347-404.
    """
    if theta_s > 0:
        Csur = (1 - np.cosh(theta_s * sigma_w)) / (np.cosh(theta_s) - 1)
    else:
        Csur = -(sigma_w**2)
    if theta_b > 0:
        return (np.exp(theta_b * Csur) - 1) / (1 - np.exp(-theta_b))
    return Csur


def _Cs_w_Vstretching_4(NZ: int, theta_s: float, theta_b: float) -> npt.NDArray[np.float64]:
    """Build `Cs_w` using ROMS Vstretching=4 ("double stretching"), with linear sigma.

    See `_C_double_stretching` for the formula and reference.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels.
    theta_s : float
        Surface control parameter.
    theta_b : float
        Bottom control parameter.

    Returns
    -------
    npt.NDArray[np.float64]
        `Cs_w`, tabulated at the `NZ + 1` w points.
    """
    return _C_double_stretching(_sigma_w_array(NZ, "linear"), theta_s, theta_b)


def _Cs_w_Vstretching_5(NZ: int, theta_s: float, theta_b: float) -> npt.NDArray[np.float64]:
    """Build `Cs_w` using ROMS Vstretching=5 (Souza et al. 2015), with quadratic sigma.

    Reuses Vstretching=4's `C(s)` formula exactly (see `_C_double_stretching`), fed the quadratic
    `sigma_w` (see `_sigma_coordinate_quadratic`) instead of the linear one.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels.
    theta_s : float
        Surface control parameter.
    theta_b : float
        Bottom control parameter.

    Returns
    -------
    npt.NDArray[np.float64]
        `Cs_w`, tabulated at the `NZ + 1` w points.
    """
    return _C_double_stretching(_sigma_w_array(NZ, "quadratic"), theta_s, theta_b)


#################################################
### monotone cubic Hermite interpolation      ###
#################################################


@numba.njit(nogil=True, fastmath=True)
def _hermite_cubic_coefficients(f0: float, f1: float, d0: float, d1: float) -> tuple[float, float, float, float]:
    """Monomial coefficients of the cubic Hermite polynomial on a unit-width cell.

    Parameters
    ----------
    f0 : float
        The value at the cell's lower node (`t=0`).
    f1 : float
        The value at the cell's upper node (`t=1`).
    d0 : float
        The derivative (with respect to the array index) at the cell's lower node.
    d1 : float
        The derivative (with respect to the array index) at the cell's upper node.

    Returns
    -------
    tuple[float, float, float, float]
        `(a, b, c, d)` such that `f(t) = a*t**3 + b*t**2 + c*t + d` matches `f0`, `f1`, `d0`, `d1`
        at `t=0`/`t=1`.
    """
    a = 2.0 * f0 + d0 - 2.0 * f1 + d1
    b = -3.0 * f0 - 2.0 * d0 + 3.0 * f1 - d1
    return a, b, d0, f0


@numba.njit(nogil=True, fastmath=True)
def _cubic_eval(a: float, b: float, c: float, d: float, t: float) -> float:
    """Evaluate `a*t**3 + b*t**2 + c*t + d` by Horner's method.

    Parameters
    ----------
    a : float
        The cubic coefficient.
    b : float
        The quadratic coefficient.
    c : float
        The linear coefficient.
    d : float
        The constant coefficient.
    t : float
        The point at which to evaluate the cubic.

    Returns
    -------
    float
        The cubic's value at `t`.
    """
    return ((a * t + b) * t + c) * t + d


@numba.njit(nogil=True, fastmath=True)
def _cubic_eval_with_derivatives(a: float, b: float, c: float, d: float, t: float) -> tuple[float, float, float]:
    """Evaluate a cubic and its first two derivatives at `t`, by Horner's method.

    Parameters
    ----------
    a : float
        The cubic coefficient.
    b : float
        The quadratic coefficient.
    c : float
        The linear coefficient.
    d : float
        The constant coefficient.
    t : float
        The point at which to evaluate the cubic.

    Returns
    -------
    tuple[float, float, float]
        The cubic's value, first derivative, and second derivative at `t`.
    """
    value = ((a * t + b) * t + c) * t + d
    first_derivative = (3.0 * a * t + 2.0 * b) * t + c
    second_derivative = 6.0 * a * t + 2.0 * b
    return value, first_derivative, second_derivative


@numba.njit(nogil=True, fastmath=True)
def _hermite_interpolation_particle(
    value_array: npt.NDArray[np.float64],
    derivative_array: npt.NDArray[np.float64],
    offset_idx: float,
    max_idx: int,
) -> float:
    """Cubic Hermite interpolation of `value_array` (with matching `derivative_array`) at `offset_idx`.

    Parameters
    ----------
    value_array : npt.NDArray[np.float64]
        The tabulated node values.
    derivative_array : npt.NDArray[np.float64]
        The tabulated node derivatives (with respect to the array index), same length as
        `value_array`.
    offset_idx : float
        The continuous index at which to interpolate.
    max_idx : int
        The largest valid lower-node index (`len(value_array) - 2`); `offset_idx` is clamped so
        the resulting lower node and its neighbour both stay in bounds.

    Returns
    -------
    float
        The interpolated value.
    """
    I0 = int(offset_idx)
    if I0 < 0:
        I0 = 0
    elif I0 > max_idx:
        I0 = max_idx
    t = offset_idx - I0
    a, b, c, d = _hermite_cubic_coefficients(
        value_array[I0], value_array[I0 + 1], derivative_array[I0], derivative_array[I0 + 1]
    )
    return _cubic_eval(a, b, c, d, t)


def _bessel_node_derivative_array(f: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Unfiltered Bessel (three-point) node derivatives, for uniformly unit-spaced nodes.

    Interior nodes use the arithmetic mean of the two adjacent secant slopes; boundary nodes use
    the one-sided 3-point formula `(3*delta_near - delta_far) / 2`. Both are classical
    finite-difference identities that are exact for any polynomial of degree at most 2, at every
    node, regardless of the polynomial's behavior between nodes - this is what lets
    `_node_derivative_array` reproduce `_sigma_coordinate_linear`/`_sigma_coordinate_quadratic`
    exactly wherever `_hyman_filter_node_derivative_array` doesn't clamp it.

    Parameters
    ----------
    f : npt.NDArray[np.float64]
        Node values, uniformly unit-spaced in index space.

    Returns
    -------
    npt.NDArray[np.float64]
        The unfiltered node derivatives, same length as `f`.
    """
    delta = np.diff(f)
    derivative = np.empty_like(f)
    derivative[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    derivative[0] = (3.0 * delta[0] - delta[1]) / 2.0
    derivative[-1] = (3.0 * delta[-1] - delta[-2]) / 2.0
    return derivative


def _hyman_filter_node_derivative_array(
    f: npt.NDArray[np.float64], raw_derivative: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Apply the Hyman (1983) monotonicity filter to unfiltered node derivatives.

    Clamps `raw_derivative` so the resulting piecewise cubic Hermite interpolant never overshoots
    or dips between nodes, given that `f` is strictly increasing (both `sigma_w` and `Cs_w` satisfy
    this contract) and the nodes are uniformly unit-spaced in index space. Interior nodes are
    capped at 3 times the smaller of the two adjacent secant slopes; boundary nodes are clamped to
    0 if their sign disagrees with the adjacent secant, otherwise capped at 3 times that secant.
    The general algorithm's opposite-sign-secants branch (which would also clamp interior nodes to
    0) never triggers here, since `f`'s secants are always positive, and is omitted.

    Parameters
    ----------
    f : npt.NDArray[np.float64]
        Node values, uniformly unit-spaced in index space. Must be strictly increasing.
    raw_derivative : npt.NDArray[np.float64]
        Unfiltered node derivatives, e.g. from `_bessel_node_derivative_array`.

    Returns
    -------
    npt.NDArray[np.float64]
        The filtered node derivatives, same length as `f`.

    References
    ----------
    Hyman, J.M., 1983: Accurate Monotonicity Preserving Cubic Interpolation. SIAM J. Sci. Stat.
    Comput., 4, 645-654.
    """
    delta = np.diff(f)
    derivative = np.empty_like(f)
    derivative[1:-1] = np.minimum(raw_derivative[1:-1], 3.0 * np.minimum(delta[:-1], delta[1:]))
    derivative[0] = min(raw_derivative[0], 3.0 * delta[0]) if np.sign(raw_derivative[0]) == np.sign(delta[0]) else 0.0
    derivative[-1] = (
        min(raw_derivative[-1], 3.0 * delta[-1]) if np.sign(raw_derivative[-1]) == np.sign(delta[-1]) else 0.0
    )
    return derivative


def _node_derivative_array(f: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Monotone cubic Hermite node derivatives for `f` (Bessel slope, Hyman-filtered).

    Used identically for both `sigma_w` and `Cs_w`: operates purely on the tabulated node values,
    with no knowledge of which formula (if any) produced them. See `_bessel_node_derivative_array`
    and `_hyman_filter_node_derivative_array` for the two steps.

    Parameters
    ----------
    f : npt.NDArray[np.float64]
        Node values, uniformly unit-spaced in index space. Must be strictly increasing.

    Returns
    -------
    npt.NDArray[np.float64]
        The node derivatives, same length as `f`.
    """
    return _hyman_filter_node_derivative_array(f, _bessel_node_derivative_array(f))


def _cubic_zidx_from_S_factory(halley_iterations: int):
    """Build a `_cubic_zidx_from_S` function with `halley_iterations` fixed as a compile-time constant.

    Called once per `CubicVerticalTransform.compute_zidx_kernel` build (not per particle), so the
    returned function is freshly compiled per distinct `halley_iterations` value, closed over
    exactly like `NZ`, `hc`, and the other values baked into that kernel.

    Parameters
    ----------
    halley_iterations : int
        The number of Halley iterations to run (see `CubicVerticalTransform`'s docstring).

    Returns
    -------
    Callable
        A `@numba.njit` function `(S, tau, sigma_w, sigma_w_derivative, C, C_derivative) -> float`
        giving the fractional vertical index, cubic-Hermite interpolating `S` within the
        bracketing w-point cell. The bracketing cell is located exactly (via
        `_compute_widx_from_S`, scheme-agnostic); the fractional position within it is then
        refined from the exact linear estimate by `halley_iterations` Halley iterations, since `S`
        has no closed-form inverse on a cubic cell.
    """
    halley_iterations_ = int(halley_iterations)

    @numba.njit(nogil=True, fastmath=True)
    def _cubic_zidx_from_S(
        S: float,
        tau: float,
        sigma_w: npt.NDArray[np.float64],
        sigma_w_derivative: npt.NDArray[np.float64],
        C: npt.NDArray[np.float64],
        C_derivative: npt.NDArray[np.float64],
    ) -> float:
        widx = _compute_widx_from_S(S, tau, sigma_w, C)

        S_low = _S_coordinate(tau, sigma_w[widx], C[widx])
        S_high = _S_coordinate(tau, sigma_w[widx + 1], C[widx + 1])
        t = (S - S_low) / (S_high - S_low)

        a_sigma, b_sigma, c_sigma, d_sigma = _hermite_cubic_coefficients(
            sigma_w[widx], sigma_w[widx + 1], sigma_w_derivative[widx], sigma_w_derivative[widx + 1]
        )
        a_C, b_C, c_C, d_C = _hermite_cubic_coefficients(
            C[widx], C[widx + 1], C_derivative[widx], C_derivative[widx + 1]
        )
        a = tau * a_sigma + (1.0 - tau) * a_C
        b = tau * b_sigma + (1.0 - tau) * b_C
        c = tau * c_sigma + (1.0 - tau) * c_C
        d = tau * d_sigma + (1.0 - tau) * d_C - S

        for _ in range(halley_iterations_):
            g, g_prime, g_double_prime = _cubic_eval_with_derivatives(a, b, c, d, t)
            t = t - 2.0 * g * g_prime / (2.0 * g_prime * g_prime - g * g_double_prime)

        return widx - _W_OFFSET + t

    return _cubic_zidx_from_S


def _compute_zidx_cubic_factory(halley_iterations: int):
    """Build a `compute_zidx_cubic` function with `halley_iterations` fixed as a compile-time constant.

    Parameters
    ----------
    halley_iterations : int
        The number of Halley iterations to run (see `CubicVerticalTransform`'s docstring).

    Returns
    -------
    Callable
        A `@numba.njit` function `(z, h, zeta, tau, sigma_w, sigma_w_derivative, C, C_derivative)
        -> float` computing the vertical index from the physical vertical coordinate, via cubic
        Hermite interpolation.
    """
    cubic_zidx_from_S = _cubic_zidx_from_S_factory(halley_iterations)

    @numba.njit(nogil=True, fastmath=True)
    def compute_zidx_cubic(
        z: float,
        h: float,
        zeta: float,
        tau: float,
        sigma_w: npt.NDArray[np.float64],
        sigma_w_derivative: npt.NDArray[np.float64],
        C: npt.NDArray[np.float64],
        C_derivative: npt.NDArray[np.float64],
    ) -> float:
        S = _S_from_z(z, h, zeta)
        return cubic_zidx_from_S(S, tau, sigma_w, sigma_w_derivative, C, C_derivative)

    return compute_zidx_cubic


#####################################################
### VerticalTransform (abstract base class)      ###
#####################################################


@dataclass(frozen=True, eq=False)
class VerticalTransform(abc.ABC):
    r"""Abstract base class for ROMS vertical coordinate transforms.

    The physical vertical position :math:`z` is

    .. math::

        z = \zeta + (\zeta + h) S

    where :math:`\zeta` is the free surface elevation and :math:`h` is the bathymetric depth. The
    nondimensional S-coordinate :math:`S` is

    .. math::

        S = \tau \sigma + (1 - \tau) C_s(\sigma)

    where :math:`C_s` is the stretching function and :math:`\tau` depends on `Vtransform` and the
    critical depth :math:`h_c`: :math:`\tau = h_c/h` for `Vtransform=1`, or
    :math:`\tau = h_c/(h_c+h)` for `Vtransform=2`. Both :math:`\sigma` and :math:`C_s` are defined
    only at the `NZ + 1` w-point sigma values - `Cs_w` gives :math:`C_s` directly (matching ROMS'
    own `Cs_w` output variable), while :math:`\sigma` is tabulated internally from
    `sigma_method`: linearly, :math:`\sigma_w(j) = j / NZ - 1` (ROMS Vstretching 1-4), or
    quadratically (ROMS Vstretching=5, see `_sigma_coordinate_quadratic`). A valid stretching
    function satisfies :math:`C_s(-1) = -1` and :math:`C_s(0) = 0`. Concrete subclasses define how
    `S` (and therefore `z`/`zidx`) is computed between these tabulated w points.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels. Must be at least 2.
    hc : float
        The critical depth. Must be strictly positive.
    Cs_w : npt.ArrayLike
        The stretching function at w points, with exactly `NZ + 1` points. Must be strictly
        increasing, with `Cs_w[0]` equal to -1 and `Cs_w[-1]` equal to 0 (to within floating-point
        tolerance).
    sigma_method : {"linear", "quadratic"}, optional
        Whether the sigma coordinate is tabulated at the w points using the linear formula (ROMS
        Vstretching 1-4) or the quadratic formula (ROMS Vstretching 5). Defaults to "linear".
    Vtransform : {1, 2}, optional
        The ROMS vertical transform, determining how :math:`\tau` is computed from :math:`h_c`
        and :math:`h`. Defaults to 2.

    Raises
    ------
    ValueError
        If `NZ` is less than 2.
        If `hc` is not strictly positive.
        If `Cs_w` is not one-dimensional.
        If `Cs_w` does not have exactly `NZ + 1` points.
        If `Cs_w` is not strictly increasing.
        If `Cs_w[0]` is not equal to -1 (to within floating-point tolerance).
        If `Cs_w[-1]` is not equal to 0 (to within floating-point tolerance).
        If `sigma_method` is not "linear" or "quadratic".
        If `Vtransform` is not 1 or 2.

    Notes
    -----
    Two instances are equal if they are the same concrete type and have equal values for every
    dataclass field (including any added by a subclass). Instances are not hashable.
    """

    NZ: int
    hc: float
    Cs_w: npt.NDArray[np.float64]
    sigma_method: Literal["linear", "quadratic"] = "linear"
    Vtransform: Literal[1, 2] = 2
    _kernel_cache: dict[tuple[str, bool], KernelFunction] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.NZ < 2:
            raise ValueError(f"NZ must be at least 2, got {self.NZ}.")
        if self.hc <= 0:
            raise ValueError(f"hc must be strictly positive, got {self.hc}.")
        if self.sigma_method not in ("linear", "quadratic"):
            raise ValueError(f"Invalid sigma_method: {self.sigma_method!r}. Must be 'linear' or 'quadratic'.")
        if self.Vtransform not in (1, 2):
            raise ValueError(f"Invalid Vtransform: {self.Vtransform!r}. Must be 1 or 2.")

        # validate Cs_w array
        Cs_w = np.array(self.Cs_w, dtype=np.float64, copy=True)
        if Cs_w.ndim != 1:
            raise ValueError(f"Cs_w must be one-dimensional, got shape {Cs_w.shape}.")
        if Cs_w.shape[0] != self.NZ + 1:
            raise ValueError(f"Cs_w must have exactly NZ+1={self.NZ + 1} points, got {Cs_w.shape[0]}.")
        if not np.all(np.diff(Cs_w) > 0):
            raise ValueError("Cs_w must be strictly increasing.")
        if not np.isclose(Cs_w[0], -1.0):
            raise ValueError(f"Cs_w[0] must equal -1.0 (the seafloor), got {Cs_w[0]}.")
        if not np.isclose(Cs_w[-1], 0.0):
            raise ValueError(f"Cs_w[-1] must equal 0.0 (the free surface), got {Cs_w[-1]}.")

        # save attributes
        Cs_w.setflags(write=False)
        object.__setattr__(self, "NZ", int(self.NZ))
        object.__setattr__(self, "hc", float(self.hc))
        object.__setattr__(self, "Cs_w", Cs_w)

    def __eq__(self, other: object) -> bool:
        """Compare every dataclass field: NZ, hc, Cs_w, sigma_method, Vtransform, and any subclass fields.

        Returns
        -------
        bool
            Whether `other` is the same concrete type as `self` with equal values for every
            comparable dataclass field.
        """
        if type(other) is not type(self):
            return NotImplemented
        for f in fields(self):
            if not f.compare:
                continue
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, np.ndarray):
                if not np.array_equal(a, b):
                    return False
            elif a != b:
                return False
        return True

    __hash__ = None  # defining __eq__ makes instances unhashable

    @abc.abstractmethod
    def compute_z_kernel(self, only_initialising: bool = False) -> KernelFunction:
        """Return a kernel function that computes the physical vertical position `z` from `zidx`.

        Parameters
        ----------
        only_initialising : bool, optional
            If False (default), compute for any active particle.
            If True, compute only for particles with status ``Status.INITIALISING``.

        Returns
        -------
        KernelFunction
            Cached on this instance: repeated calls with the same `only_initialising` return the
            same compiled function without recompiling.
        """
        ...

    @abc.abstractmethod
    def compute_zidx_kernel(self, only_initialising: bool = False) -> KernelFunction:
        """Return a kernel function that computes the vertical index `zidx` from `z`.

        Parameters
        ----------
        only_initialising : bool, optional
            If False (default), compute for any active particle.
            If True, compute only for particles with status ``Status.INITIALISING``.

        Returns
        -------
        KernelFunction
            Cached on this instance: repeated calls with the same `only_initialising` return the
            same compiled function without recompiling.
        """
        ...

    @classmethod
    def from_Vstretching_1(cls, NZ: int, hc: float, theta_s: float, theta_b: float, **kwargs) -> Self:
        r"""Construct from ROMS Vstretching=1 (Song & Haidvogel 1994), with linear sigma.

        .. math::

            C_s(\sigma) = (1 - \theta_b) \frac{\sinh(\theta_s \sigma)}{\sinh(\theta_s)}
            + \theta_b \left[ -0.5 + 0.5 \frac{\tanh(\theta_s(\sigma + 0.5))}{\tanh(0.5\theta_s)} \right]

        with :math:`C_s(\sigma) = \sigma` if :math:`\theta_s = 0`. Valid for
        :math:`0 < \theta_s \le 8`, :math:`0 \le \theta_b \le 1`.

        Parameters
        ----------
        NZ : int
            The total number of vertical rho levels.
        hc : float
            The critical depth.
        theta_s : float
            Surface control parameter.
        theta_b : float
            Bottom control parameter.
        **kwargs
            Forwarded to the concrete subclass's constructor (e.g. `horizontal_stencil_half_width`,
            `Vtransform`).

        Returns
        -------
        Self
            A new instance with `sigma_method="linear"` and `Cs_w` built from `theta_s`/`theta_b`.

        References
        ----------
        Song, Y. and D. Haidvogel, 1994: A semi-implicit ocean circulation model using a
        generalized topography-following coordinate system. J. Comp. Physics, 115, 228-244.
        """
        return cls(NZ=NZ, hc=hc, Cs_w=_Cs_w_Vstretching_1(NZ, theta_s, theta_b), sigma_method="linear", **kwargs)

    @classmethod
    def from_Vstretching_2(cls, NZ: int, hc: float, theta_s: float, theta_b: float, **kwargs) -> Self:
        r"""Construct from ROMS Vstretching=2 (Shchepetkin & McWilliams 2005), with linear sigma.

        .. math::

            C_{sur}(\sigma) = \frac{1 - \cosh(\theta_s \sigma)}{\cosh(\theta_s) - 1}

        if :math:`\theta_s > 0`, else :math:`C_{sur}(\sigma) = \sigma`. If :math:`\theta_b > 0`,
        blended with a bottom-intensifying function:

        .. math::

            C_{bot}(\sigma) = \frac{\sinh(\theta_b(\sigma+1))}{\sinh(\theta_b)} - 1 \qquad
            C_{weight} = (\sigma+1)\left[2-(\sigma+1)\right] \qquad
            C_s(\sigma) = C_{weight} C_{sur}(\sigma) + (1 - C_{weight}) C_{bot}(\sigma)

        otherwise :math:`C_s(\sigma) = C_{sur}(\sigma)`. Unlike Vstretching=1, satisfies
        :math:`dC_s/d\sigma \to 0` as :math:`\sigma \to 0`.

        Parameters
        ----------
        NZ : int
            The total number of vertical rho levels.
        hc : float
            The critical depth.
        theta_s : float
            Surface control parameter.
        theta_b : float
            Bottom control parameter.
        **kwargs
            Forwarded to the concrete subclass's constructor (e.g. `horizontal_stencil_half_width`,
            `Vtransform`).

        Returns
        -------
        Self
            A new instance with `sigma_method="linear"` and `Cs_w` built from `theta_s`/`theta_b`.

        References
        ----------
        Shchepetkin, A.F. and J.C. McWilliams, 2005: The regional oceanic modeling system
        (ROMS): a split-explicit, free-surface, topography-following-coordinate oceanic model.
        Ocean Modelling, 9, 347-404.
        """
        return cls(NZ=NZ, hc=hc, Cs_w=_Cs_w_Vstretching_2(NZ, theta_s, theta_b), sigma_method="linear", **kwargs)

    @classmethod
    def from_Vstretching_3(cls, NZ: int, hc: float, theta_s: float, theta_b: float, **kwargs) -> Self:
        r"""Construct from ROMS Vstretching=3 (R. Geyer), with linear sigma.

        Designed for high bottom boundary-layer resolution in shallow coastal applications.
        `theta_s`/`theta_b` are reinterpreted as surface/bottom stretching exponents
        :math:`\alpha`/:math:`\beta`:

        .. math::

            C_{sur}(\sigma) = -\frac{\ln\cosh\left(H|\sigma|^\alpha\right)}{\ln\cosh(H)} \qquad
            C_{bot}(\sigma) = \frac{\ln\cosh\left(H(\sigma+1)^\beta\right)}{\ln\cosh(H)} - 1 \qquad
            C_{weight} = 0.5\left[1 - \tanh\left(H(\sigma+0.5)\right)\right]

        .. math::

            C_s(\sigma) = C_{weight} C_{bot}(\sigma) + (1 - C_{weight}) C_{sur}(\sigma)

        with :math:`H = 3.0` fixed. :math:`\alpha`: 0.65 (minimal) to 1.0 (significant) surface
        stretching. :math:`\beta`: 0.58 (none) to 3.0 (super-high) bottom stretching.

        Parameters
        ----------
        NZ : int
            The total number of vertical rho levels.
        hc : float
            The critical depth.
        theta_s : float
            Surface stretching exponent (alpha).
        theta_b : float
            Bottom stretching exponent (beta).
        **kwargs
            Forwarded to the concrete subclass's constructor (e.g. `horizontal_stencil_half_width`,
            `Vtransform`).

        Returns
        -------
        Self
            A new instance with `sigma_method="linear"` and `Cs_w` built from `theta_s`/`theta_b`.
        """
        return cls(NZ=NZ, hc=hc, Cs_w=_Cs_w_Vstretching_3(NZ, theta_s, theta_b), sigma_method="linear", **kwargs)

    @classmethod
    def from_Vstretching_4(cls, NZ: int, hc: float, theta_s: float, theta_b: float, **kwargs) -> Self:
        r"""Construct from ROMS Vstretching=4 ("double stretching"), with linear sigma.

        .. math::

            C_{sur}(\sigma) = \frac{1 - \cosh(\theta_s \sigma)}{\cosh(\theta_s) - 1}

        if :math:`\theta_s > 0`, else :math:`C_{sur}(\sigma) = -\sigma^2`. If
        :math:`\theta_b > 0`:

        .. math::

            C_s(\sigma) = \frac{\exp(\theta_b C_{sur}(\sigma)) - 1}{1 - \exp(-\theta_b)}

        otherwise :math:`C_s(\sigma) = C_{sur}(\sigma)`. Valid for :math:`0 < \theta_s \le 10`,
        :math:`0 \le \theta_b \le 3`. Continuous in both parameters as either tends to zero.

        Parameters
        ----------
        NZ : int
            The total number of vertical rho levels.
        hc : float
            The critical depth.
        theta_s : float
            Surface control parameter.
        theta_b : float
            Bottom control parameter.
        **kwargs
            Forwarded to the concrete subclass's constructor (e.g. `horizontal_stencil_half_width`,
            `Vtransform`).

        Returns
        -------
        Self
            A new instance with `sigma_method="linear"` and `Cs_w` built from `theta_s`/`theta_b`.

        References
        ----------
        Shchepetkin, A.F. and J.C. McWilliams, 2005: The regional oceanic modeling system
        (ROMS): a split-explicit, free-surface, topography-following-coordinate oceanic model.
        Ocean Modelling, 9, 347-404.
        """
        return cls(NZ=NZ, hc=hc, Cs_w=_Cs_w_Vstretching_4(NZ, theta_s, theta_b), sigma_method="linear", **kwargs)

    @classmethod
    def from_Vstretching_5(cls, NZ: int, hc: float, theta_s: float, theta_b: float, **kwargs) -> Self:
        r"""Construct from ROMS Vstretching=5 (Souza et al. 2015), with quadratic sigma.

        Reuses Vstretching=4's `C(s)` formula exactly (see `from_Vstretching_4`), but fed the
        quadratic sigma coordinate instead of the linear one:

        .. math::

            \sigma_w(k) = -\frac{k^2 - 2kN + k + N^2 - N}{N^2 - N} - 0.01 \frac{k^2 - kN}{1 - N}

        for `k = 0, ..., N`. Designed to enhance the surface exchange layer by concentrating more
        levels near the surface than the linear sigma coordinate does.

        Parameters
        ----------
        NZ : int
            The total number of vertical rho levels.
        hc : float
            The critical depth.
        theta_s : float
            Surface control parameter (as for Vstretching=4).
        theta_b : float
            Bottom control parameter (as for Vstretching=4).
        **kwargs
            Forwarded to the concrete subclass's constructor (e.g. `horizontal_stencil_half_width`,
            `Vtransform`).

        Returns
        -------
        Self
            A new instance with `sigma_method="quadratic"` and `Cs_w` built from
            `theta_s`/`theta_b`.

        References
        ----------
        Souza, J., B.S. Powell, A.C. Castillo-Trujillo, and P. Flament, 2015: The Vorticity
        Balance of the Ocean Surface in Hawaii from a Regional Reanalysis. J. Phys. Oceanogr.,
        45, 424-440.
        """
        return cls(NZ=NZ, hc=hc, Cs_w=_Cs_w_Vstretching_5(NZ, theta_s, theta_b), sigma_method="quadratic", **kwargs)


def _validate_horizontal_stencil_half_width(value: int) -> int:
    """Validate and coerce `horizontal_stencil_half_width`.

    Shared by every `VerticalTransform` subclass that offers a configurable horizontal
    interpolation stencil.

    Parameters
    ----------
    value : int
        The proposed half-width.

    Returns
    -------
    int
        `value`, coerced to `int`.

    Raises
    ------
    ValueError
        If `value` is not a positive integer.
    """
    if value < 1:
        raise ValueError(f"horizontal_stencil_half_width must be a positive integer, got {value}.")
    return int(value)


def _validate_halley_iterations(value: int) -> int:
    """Validate and coerce `CubicVerticalTransform.halley_iterations`.

    Parameters
    ----------
    value : int
        The proposed iteration count.

    Returns
    -------
    int
        `value`, coerced to `int`.

    Raises
    ------
    ValueError
        If `value` is not a positive integer.
    """
    if value < 1:
        raise ValueError(f"halley_iterations must be a positive integer, got {value}.")
    return int(value)


#####################################################
### LinearVerticalTransform                      ###
#####################################################


@dataclass(frozen=True, eq=False)
class LinearVerticalTransform(VerticalTransform):
    r"""ROMS vertical transform, with sigma and the stretching function interpolated linearly.

    `S` (and therefore `z`) is computed by linearly interpolating between the tabulated w-point
    values of `sigma` and `Cs_w` - see `VerticalTransform` for the shared S-coordinate/z-coordinate
    transform and the `NZ`/`hc`/`Cs_w`/`sigma_method`/`Vtransform` parameters.

    For `Vtransform=1`, `hc` must not exceed the shallowest bathymetric depth anywhere particles
    are tracked (`hc <= hmin`), a well-known ROMS constraint: `tau = hc/h` otherwise exceeds 1
    somewhere in the domain, and the blend `S = tau*sigma + (1-tau)*C` of two independently
    strictly-increasing sequences is no longer guaranteed increasing *between* grid points, which
    the binary search in `compute_zidx_kernel` assumes.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels. Must be at least 2.
    hc : float
        The critical depth. Must be strictly positive.
    Cs_w : npt.ArrayLike
        The stretching function at w points, with exactly `NZ + 1` points. Must be strictly
        increasing, with `Cs_w[0]` equal to -1 and `Cs_w[-1]` equal to 0 (to within floating-point
        tolerance).
    sigma_method : {"linear", "quadratic"}, optional
        Whether the sigma coordinate is tabulated at the w points using the linear formula (ROMS
        Vstretching 1-4) or the quadratic formula (ROMS Vstretching 5). Defaults to "linear".
    Vtransform : {1, 2}, optional
        The ROMS vertical transform. Defaults to 2.
    horizontal_stencil_half_width : int, optional
        The half-width `N` of the Lagrange interpolation stencil used for `h` and `zeta` (`2N`
        points in each horizontal dimension). Defaults to 1, i.e. bilinear interpolation.

    Raises
    ------
    ValueError
        If `NZ` is less than 2.
        If `hc` is not strictly positive.
        If `horizontal_stencil_half_width` is not a positive integer.
        If `Cs_w` is not one-dimensional.
        If `Cs_w` does not have exactly `NZ + 1` points.
        If `Cs_w` is not strictly increasing.
        If `Cs_w[0]` is not equal to -1 (to within floating-point tolerance).
        If `Cs_w[-1]` is not equal to 0 (to within floating-point tolerance).
        If `sigma_method` is not "linear" or "quadratic".
        If `Vtransform` is not 1 or 2.

    Notes
    -----
    Two instances are equal if they have the same `NZ`, `hc`, `Cs_w`, `sigma_method`,
    `Vtransform`, and `horizontal_stencil_half_width`. Instances are not hashable.
    """

    horizontal_stencil_half_width: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "horizontal_stencil_half_width",
            _validate_horizontal_stencil_half_width(self.horizontal_stencil_half_width),
        )

    def compute_z_kernel(self, only_initialising: bool = False) -> KernelFunction:
        """Return a kernel function that computes the physical vertical position `z` from `zidx`.

        Parameters
        ----------
        only_initialising : bool, optional
            If False (default), compute for any active particle.
            If True, compute only for particles with status ``Status.INITIALISING``.

        Returns
        -------
        KernelFunction
            Cached on this instance: repeated calls with the same `only_initialising` return the
            same compiled function without recompiling.
        """
        key = ("z", bool(only_initialising))
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _linear_vertical_transform_compute_z_kernel(
                self.NZ,
                self.hc,
                self.Cs_w,
                self.sigma_method,
                self.Vtransform,
                self.horizontal_stencil_half_width,
                only_initialising,
            )
        return self._kernel_cache[key]

    def compute_zidx_kernel(self, only_initialising: bool = False) -> KernelFunction:
        """Return a kernel function that computes the vertical index `zidx` from `z`.

        Parameters
        ----------
        only_initialising : bool, optional
            If False (default), compute for any active particle.
            If True, compute only for particles with status ``Status.INITIALISING``.

        Returns
        -------
        KernelFunction
            Cached on this instance: repeated calls with the same `only_initialising` return the
            same compiled function without recompiling.
        """
        key = ("zidx", bool(only_initialising))
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _linear_vertical_transform_compute_zidx_kernel(
                self.NZ,
                self.hc,
                self.Cs_w,
                self.sigma_method,
                self.Vtransform,
                self.horizontal_stencil_half_width,
                only_initialising,
            )
        return self._kernel_cache[key]


def _linear_vertical_transform_compute_z_kernel(
    NZ: int,
    hc: float,
    C: npt.NDArray[np.float64],
    sigma_method: str,
    Vtransform: int,
    horizontal_stencil_half_width: int,
    only_initialising: bool,
) -> KernelFunction:
    """Build the compiled kernel function backing `LinearVerticalTransform.compute_z_kernel`.

    `C` and the sigma coordinate tabulated from `sigma_method` are both captured by closure and
    linearly interpolated at each particle's continuous `zidx`.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels. Baked into the compiled kernel as a compile-time
        constant (via the tabulated `sigma_w` array it determines the shape of).
    hc : float
        The critical depth. Baked into the compiled kernel as a compile-time constant.
    C : npt.NDArray[np.float64]
        `Cs_w`, the complete stretching function array at w points, with exactly `NZ + 1` points.
        Captured by closure.
    sigma_method : str
        Either "linear" or "quadratic"; determines the tabulated sigma array captured by closure.
    Vtransform : int
        Either 1 or 2; determines the `tau(hc, h)` function captured by closure.
    horizontal_stencil_half_width : int
        The half-width `N` of the Lagrange interpolation stencil used for `h` and `zeta`.
    only_initialising : bool
        If False, compute for any active particle. If True, compute only for particles with
        status ``Status.INITIALISING``.

    Returns
    -------
    KernelFunction
        A kernel function that computes the physical vertical position `z`, specialized for the
        given `NZ`, `hc`, `C`, `sigma_method`, `Vtransform`, `horizontal_stencil_half_width`, and
        `only_initialising`.
    """
    hc_: np.float64 = np.float64(hc)
    C_: npt.NDArray[np.float64] = np.array(C, dtype=np.float64)
    C_max_idx = C_.shape[0] - 2
    sigma_w_: npt.NDArray[np.float64] = _sigma_w_array(NZ, sigma_method)
    sigma_max_idx = sigma_w_.shape[0] - 2
    tau_fn = _tau_fn_for_Vtransform(Vtransform)
    N = int(horizontal_stencil_half_width)
    horizontal_interpolation_particle = lagrange_2D_particle_factory(N, N)
    only_initialising_: bool = bool(only_initialising)

    @kernel_function(
        particle_property_keys=["status", "zidx", "yidx", "xidx", "z"],
        field_data_keys=["h", "zeta"],
    )
    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def compute_z_kernel_function(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        h_array: npt.NDArray[np.float64],
        h_offy: float,
        h_offx: float,
        zeta_array: npt.NDArray[np.float64],
        zeta_offy: float,
        zeta_offx: float,
    ) -> None:
        """Compute the vertical position 'z' of particles in ROMS vertical coordinates.

        Parameters
        ----------
        status : npt.NDArray[np.uint8]
            The particle property array of statuses.
        zidx : npt.NDArray[np.float64]
            The particle property array of vertical indices.
        yidx : npt.NDArray[np.float64]
            The particle property array of y indices.
        xidx : npt.NDArray[np.float64]
            The particle property array of x indices.
        z : npt.NDArray[np.float64]
            The particle property array to store the computed physical vertical position.
        h_array : npt.NDArray[np.float64]
            The bathymetry field array.
        h_offy : float
            The y offset for the bathymetry field.
        h_offx : float
            The x offset for the bathymetry field.
        zeta_array : npt.NDArray[np.float64]
            The free surface field array.
        zeta_offy : float
            The y offset for the free surface field.
        zeta_offx : float
            The x offset for the free surface field.

        Raises
        ------
        ValueError
            If the h or zeta field arrays do not have at least `2 * horizontal_stencil_half_width`
            points in the relevant dimension(s) to avoid out-of-bounds memory access.
        """
        h_max_idx_0 = h_array.shape[0] - 2 * N
        h_max_idx_1 = h_array.shape[1] - 2 * N
        zeta_max_idx_0 = zeta_array.shape[0] - 2 * N
        zeta_max_idx_1 = zeta_array.shape[1] - 2 * N
        if h_max_idx_0 < 0 or h_max_idx_1 < 0:
            raise ValueError(
                "h field must have at least 2 * horizontal_stencil_half_width points in each "
                "dimension to avoid out-of-bounds memory access."
            )
        if zeta_max_idx_0 < 0 or zeta_max_idx_1 < 0:
            raise ValueError(
                "zeta field must have at least 2 * horizontal_stencil_half_width points in each "
                "dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.shape[0]):  # ty: ignore[not-iterable]
            # only_initialising_ is a compile-time constant, so this branch is optimized away at compile time
            if only_initialising_:
                if status[i] != _INITIALISING:  # only compute for initialising particles
                    continue
            else:
                if status[i] & INACTIVE_FLAG:  # only compute for active particles
                    continue

            h_value = horizontal_interpolation_particle(
                h_array, yidx[i] + h_offy, xidx[i] + h_offx, h_max_idx_0, h_max_idx_1
            )
            zeta_value = horizontal_interpolation_particle(
                zeta_array, yidx[i] + zeta_offy, xidx[i] + zeta_offx, zeta_max_idx_0, zeta_max_idx_1
            )
            tau_value = tau_fn(hc_, h_value)
            sigma_value = linear_interpolation_particle(sigma_w_, zidx[i] + _W_OFFSET, sigma_max_idx)
            C_value = linear_interpolation_particle(C_, zidx[i] + _W_OFFSET, C_max_idx)
            z[i] = compute_z(sigma_value, tau_value, h_value, C_value, zeta_value)

    return compute_z_kernel_function


def _linear_vertical_transform_compute_zidx_kernel(
    NZ: int,
    hc: float,
    C: npt.NDArray[np.float64],
    sigma_method: str,
    Vtransform: int,
    horizontal_stencil_half_width: int,
    only_initialising: bool,
) -> KernelFunction:
    """Build the compiled kernel function backing `LinearVerticalTransform.compute_zidx_kernel`.

    `C` and the sigma coordinate tabulated from `sigma_method` are both captured by closure -
    `field_data_keys` below is `["h", "zeta"]` only.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels. Baked into the compiled kernel as a compile-time
        constant (via the tabulated `sigma_w` array it determines the shape of).
    hc : float
        The critical depth. Baked into the compiled kernel as a compile-time constant.
    C : npt.NDArray[np.float64]
        `Cs_w`, the complete stretching function array at w points, with exactly `NZ + 1` points.
        Captured by closure.
    sigma_method : str
        Either "linear" or "quadratic"; determines the tabulated sigma array captured by closure.
    Vtransform : int
        Either 1 or 2; determines the `tau(hc, h)` function captured by closure.
    horizontal_stencil_half_width : int
        The half-width `N` of the Lagrange interpolation stencil used for `h` and `zeta`.
    only_initialising : bool
        If False, compute for any active particle. If True, compute only for particles with
        status ``Status.INITIALISING``.

    Returns
    -------
    KernelFunction
        A kernel function that computes the vertical index `zidx`, specialized for the given
        `NZ`, `hc`, `C`, `sigma_method`, `Vtransform`, `horizontal_stencil_half_width`, and
        `only_initialising`.
    """
    hc_: np.float64 = np.float64(hc)
    C_: npt.NDArray[np.float64] = np.array(C, dtype=np.float64)
    sigma_w_: npt.NDArray[np.float64] = _sigma_w_array(NZ, sigma_method)
    tau_fn = _tau_fn_for_Vtransform(Vtransform)
    N = int(horizontal_stencil_half_width)
    horizontal_interpolation_particle = lagrange_2D_particle_factory(N, N)
    only_initialising_: bool = bool(only_initialising)

    @kernel_function(
        particle_property_keys=["status", "zidx", "yidx", "xidx", "z"],
        field_data_keys=["h", "zeta"],
    )
    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def compute_zidx_kernel_function(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        h_array: npt.NDArray[np.float64],
        h_offy: float,
        h_offx: float,
        zeta_array: npt.NDArray[np.float64],
        zeta_offy: float,
        zeta_offx: float,
    ) -> None:
        """Compute the vertical index 'zidx' of particles in ROMS vertical coordinates.

        Parameters
        ----------
        status : npt.NDArray[np.uint8]
            The particle property array of statuses.
        zidx : npt.NDArray[np.float64]
            The particle property array to store the computed vertical index.
        yidx : npt.NDArray[np.float64]
            The particle property array of y indices.
        xidx : npt.NDArray[np.float64]
            The particle property array of x indices.
        z : npt.NDArray[np.float64]
            The particle property array of physical vertical positions.
        h_array : npt.NDArray[np.float64]
            The bathymetry field array.
        h_offy : float
            The y offset for the bathymetry field.
        h_offx : float
            The x offset for the bathymetry field.
        zeta_array : npt.NDArray[np.float64]
            The free surface field array.
        zeta_offy : float
            The y offset for the free surface field.
        zeta_offx : float
            The x offset for the free surface field.

        Raises
        ------
        ValueError
            If the h or zeta field arrays do not have at least `2 * horizontal_stencil_half_width`
            points in the relevant dimension(s) to avoid out-of-bounds memory access.
        """
        h_max_idx_0 = h_array.shape[0] - 2 * N
        h_max_idx_1 = h_array.shape[1] - 2 * N
        zeta_max_idx_0 = zeta_array.shape[0] - 2 * N
        zeta_max_idx_1 = zeta_array.shape[1] - 2 * N
        if h_max_idx_0 < 0 or h_max_idx_1 < 0:
            raise ValueError(
                "h field must have at least 2 * horizontal_stencil_half_width points in each "
                "dimension to avoid out-of-bounds memory access."
            )
        if zeta_max_idx_0 < 0 or zeta_max_idx_1 < 0:
            raise ValueError(
                "zeta field must have at least 2 * horizontal_stencil_half_width points in each "
                "dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.shape[0]):  # ty: ignore[not-iterable]
            # only_initialising_ is a compile-time constant, so this branch is optimized away at compile time
            if only_initialising_:
                if status[i] != _INITIALISING:  # only compute for initialising particles
                    continue
            else:
                if status[i] & INACTIVE_FLAG:  # only compute for active particles
                    continue

            h_value = horizontal_interpolation_particle(
                h_array, yidx[i] + h_offy, xidx[i] + h_offx, h_max_idx_0, h_max_idx_1
            )
            zeta_value = horizontal_interpolation_particle(
                zeta_array, yidx[i] + zeta_offy, xidx[i] + zeta_offx, zeta_max_idx_0, zeta_max_idx_1
            )
            tau_value = tau_fn(hc_, h_value)
            zidx[i] = compute_zidx_linear(z[i], h_value, zeta_value, tau_value, sigma_w_, C_)

    return compute_zidx_kernel_function


#####################################################
### CubicVerticalTransform                       ###
#####################################################


@dataclass(frozen=True, eq=False)
class CubicVerticalTransform(VerticalTransform):
    r"""ROMS vertical transform, with sigma and the stretching function interpolated using monotone cubic Hermite splines.

    `S` (and therefore `z`) is computed by cubic-Hermite interpolating between the tabulated
    w-point values of `sigma` and `Cs_w` - see `VerticalTransform` for the shared
    S-coordinate/z-coordinate transform and the `NZ`/`hc`/`Cs_w`/`sigma_method`/`Vtransform`
    parameters. Both arrays' node derivatives are built identically, by an unfiltered Bessel
    (three-point) slope estimate, then a Hyman (1983) monotonicity filter. This guarantees every
    cell's cubic satisfies `d(sigma)/d(zidx) >= 0` and `d(Cs_w)/d(zidx) >= 0` everywhere, which
    in turn guarantees `S` itself is monotonic everywhere in the domain -
    `dS/d(zidx) = tau*d(sigma)/d(zidx) + (1-tau)*d(Cs_w)/d(zidx)`. The Bessel slope estimate is
    exact for any polynomial of degree at most 2, at every node, regardless of its behavior
    between nodes - since both `sigma_method` formulae are such polynomials, `sigma` is reproduced
    exactly wherever the Hyman filter leaves that estimate unchanged, which holds everywhere for
    `sigma_method="linear"` and everywhere `sigma_method="quadratic"`'s underlying continuous
    formula is itself monotonic between its tabulated points; where that formula would otherwise
    dip, the filter trades exact reproduction for guaranteed monotonicity of `S`. The inverse has
    no closed form on a cubic cell and is instead computed with a fixed number of Halley iterations
    (`halley_iterations`) seeded from the exact linear-interpolation estimate. The default (3)
    converges to a residual far below any physically meaningful tolerance for typical grids, but
    is not bit-exact, and convergence slows where the seed starts far from the true root -
    notably where `tau` (`hc/h` for `Vtransform=1`, `hc/(hc+h)` for `Vtransform=2`) is close to
    its maximum of 1, combined with a small `NZ`. Increase `halley_iterations` if round-tripping
    such a grid needs more precision.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels. Must be at least 2.
    hc : float
        The critical depth. Must be strictly positive.
    Cs_w : npt.ArrayLike
        The stretching function at w points, with exactly `NZ + 1` points. Must be strictly
        increasing, with `Cs_w[0]` equal to -1 and `Cs_w[-1]` equal to 0 (to within floating-point
        tolerance).
    sigma_method : {"linear", "quadratic"}, optional
        Whether the sigma coordinate is tabulated at the w points using the linear formula (ROMS
        Vstretching 1-4) or the quadratic formula (ROMS Vstretching 5). Defaults to "linear".
    Vtransform : {1, 2}, optional
        The ROMS vertical transform. Defaults to 2.
    horizontal_stencil_half_width : int, optional
        The half-width `N` of the Lagrange interpolation stencil used for `h` and `zeta` (`2N`
        points in each horizontal dimension). Defaults to 2, i.e. bicubic interpolation.
    halley_iterations : int, optional
        The number of Halley iterations used to invert `S` on a cubic cell. Defaults to 3, which
        is sufficient for typical grids; see the class docstring above for when more may be
        needed. Baked into the compiled `compute_zidx_kernel` as a compile-time constant.

    Raises
    ------
    ValueError
        If `NZ` is less than 2.
        If `hc` is not strictly positive.
        If `horizontal_stencil_half_width` is not a positive integer.
        If `halley_iterations` is not a positive integer.
        If `Cs_w` is not one-dimensional.
        If `Cs_w` does not have exactly `NZ + 1` points.
        If `Cs_w` is not strictly increasing.
        If `Cs_w[0]` is not equal to -1 (to within floating-point tolerance).
        If `Cs_w[-1]` is not equal to 0 (to within floating-point tolerance).
        If `sigma_method` is not "linear" or "quadratic".
        If `Vtransform` is not 1 or 2.

    Notes
    -----
    Two instances are equal if they have the same `NZ`, `hc`, `Cs_w`, `sigma_method`,
    `Vtransform`, `horizontal_stencil_half_width`, and `halley_iterations`. Instances are not
    hashable.

    References
    ----------
    Fritsch, F.N. and R.E. Carlson, 1980: Monotone Piecewise Cubic Interpolation. SIAM J. Numer.
    Anal., 17, 238-246.

    Hyman, J.M., 1983: Accurate Monotonicity Preserving Cubic Interpolation. SIAM J. Sci. Stat.
    Comput., 4, 645-654.
    """

    horizontal_stencil_half_width: int = 2
    halley_iterations: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "horizontal_stencil_half_width",
            _validate_horizontal_stencil_half_width(self.horizontal_stencil_half_width),
        )
        object.__setattr__(self, "halley_iterations", _validate_halley_iterations(self.halley_iterations))

    def compute_z_kernel(self, only_initialising: bool = False) -> KernelFunction:
        """Return a kernel function that computes the physical vertical position `z` from `zidx`.

        Parameters
        ----------
        only_initialising : bool, optional
            If False (default), compute for any active particle.
            If True, compute only for particles with status ``Status.INITIALISING``.

        Returns
        -------
        KernelFunction
            Cached on this instance: repeated calls with the same `only_initialising` return the
            same compiled function without recompiling.
        """
        key = ("z", bool(only_initialising))
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _cubic_vertical_transform_compute_z_kernel(
                self.NZ,
                self.hc,
                self.Cs_w,
                self.sigma_method,
                self.Vtransform,
                self.horizontal_stencil_half_width,
                only_initialising,
            )
        return self._kernel_cache[key]

    def compute_zidx_kernel(self, only_initialising: bool = False) -> KernelFunction:
        """Return a kernel function that computes the vertical index `zidx` from `z`.

        Parameters
        ----------
        only_initialising : bool, optional
            If False (default), compute for any active particle.
            If True, compute only for particles with status ``Status.INITIALISING``.

        Returns
        -------
        KernelFunction
            Cached on this instance: repeated calls with the same `only_initialising` return the
            same compiled function without recompiling.
        """
        key = ("zidx", bool(only_initialising))
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _cubic_vertical_transform_compute_zidx_kernel(
                self.NZ,
                self.hc,
                self.Cs_w,
                self.sigma_method,
                self.Vtransform,
                self.horizontal_stencil_half_width,
                self.halley_iterations,
                only_initialising,
            )
        return self._kernel_cache[key]


def _cubic_vertical_transform_compute_z_kernel(
    NZ: int,
    hc: float,
    C: npt.NDArray[np.float64],
    sigma_method: str,
    Vtransform: int,
    horizontal_stencil_half_width: int,
    only_initialising: bool,
) -> KernelFunction:
    """Build the compiled kernel function backing `CubicVerticalTransform.compute_z_kernel`.

    `C`, the sigma coordinate tabulated from `sigma_method`, and both of their node derivatives
    are captured by closure and cubic-Hermite interpolated at each particle's continuous `zidx` -
    `field_data_keys` below is `["h", "zeta"]` only, with no `C`/`sigma` declared as field data.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels. Baked into the compiled kernel as a compile-time
        constant (via the tabulated `sigma_w` array it determines the shape of).
    hc : float
        The critical depth. Baked into the compiled kernel as a compile-time constant.
    C : npt.NDArray[np.float64]
        `Cs_w`, the complete stretching function array at w points, with exactly `NZ + 1` points.
        Captured by closure.
    sigma_method : str
        Either "linear" or "quadratic"; determines the tabulated sigma array captured by closure.
    Vtransform : int
        Either 1 or 2; determines the `tau(hc, h)` function captured by closure.
    horizontal_stencil_half_width : int
        The half-width `N` of the Lagrange interpolation stencil used for `h` and `zeta`.
    only_initialising : bool
        If False, compute for any active particle. If True, compute only for particles with
        status ``Status.INITIALISING``.

    Returns
    -------
    KernelFunction
        A kernel function that computes the physical vertical position `z`, specialized for the
        given `NZ`, `hc`, `C`, `sigma_method`, `Vtransform`, `horizontal_stencil_half_width`, and
        `only_initialising`.
    """
    hc_: np.float64 = np.float64(hc)
    C_: npt.NDArray[np.float64] = np.array(C, dtype=np.float64)
    C_max_idx = C_.shape[0] - 2
    C_derivative_: npt.NDArray[np.float64] = _node_derivative_array(C_)
    sigma_w_: npt.NDArray[np.float64] = _sigma_w_array(NZ, sigma_method)
    sigma_max_idx = sigma_w_.shape[0] - 2
    sigma_w_derivative_: npt.NDArray[np.float64] = _node_derivative_array(sigma_w_)
    tau_fn = _tau_fn_for_Vtransform(Vtransform)
    N = int(horizontal_stencil_half_width)
    horizontal_interpolation_particle = lagrange_2D_particle_factory(N, N)
    only_initialising_: bool = bool(only_initialising)

    @kernel_function(
        particle_property_keys=["status", "zidx", "yidx", "xidx", "z"],
        field_data_keys=["h", "zeta"],
    )
    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def compute_z_kernel_function(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        h_array: npt.NDArray[np.float64],
        h_offy: float,
        h_offx: float,
        zeta_array: npt.NDArray[np.float64],
        zeta_offy: float,
        zeta_offx: float,
    ) -> None:
        """Compute the vertical position 'z' of particles in ROMS vertical coordinates.

        Parameters
        ----------
        status : npt.NDArray[np.uint8]
            The particle property array of statuses.
        zidx : npt.NDArray[np.float64]
            The particle property array of vertical indices.
        yidx : npt.NDArray[np.float64]
            The particle property array of y indices.
        xidx : npt.NDArray[np.float64]
            The particle property array of x indices.
        z : npt.NDArray[np.float64]
            The particle property array to store the computed physical vertical position.
        h_array : npt.NDArray[np.float64]
            The bathymetry field array.
        h_offy : float
            The y offset for the bathymetry field.
        h_offx : float
            The x offset for the bathymetry field.
        zeta_array : npt.NDArray[np.float64]
            The free surface field array.
        zeta_offy : float
            The y offset for the free surface field.
        zeta_offx : float
            The x offset for the free surface field.

        Raises
        ------
        ValueError
            If the h or zeta field arrays do not have at least `2 * horizontal_stencil_half_width`
            points in the relevant dimension(s) to avoid out-of-bounds memory access.
        """
        h_max_idx_0 = h_array.shape[0] - 2 * N
        h_max_idx_1 = h_array.shape[1] - 2 * N
        zeta_max_idx_0 = zeta_array.shape[0] - 2 * N
        zeta_max_idx_1 = zeta_array.shape[1] - 2 * N
        if h_max_idx_0 < 0 or h_max_idx_1 < 0:
            raise ValueError(
                "h field must have at least 2 * horizontal_stencil_half_width points in each "
                "dimension to avoid out-of-bounds memory access."
            )
        if zeta_max_idx_0 < 0 or zeta_max_idx_1 < 0:
            raise ValueError(
                "zeta field must have at least 2 * horizontal_stencil_half_width points in each "
                "dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.shape[0]):  # ty: ignore[not-iterable]
            # only_initialising_ is a compile-time constant, so this branch is optimized away at compile time
            if only_initialising_:
                if status[i] != _INITIALISING:  # only compute for initialising particles
                    continue
            else:
                if status[i] & INACTIVE_FLAG:  # only compute for active particles
                    continue

            h_value = horizontal_interpolation_particle(
                h_array, yidx[i] + h_offy, xidx[i] + h_offx, h_max_idx_0, h_max_idx_1
            )
            zeta_value = horizontal_interpolation_particle(
                zeta_array, yidx[i] + zeta_offy, xidx[i] + zeta_offx, zeta_max_idx_0, zeta_max_idx_1
            )
            tau_value = tau_fn(hc_, h_value)
            sigma_value = _hermite_interpolation_particle(
                sigma_w_, sigma_w_derivative_, zidx[i] + _W_OFFSET, sigma_max_idx
            )
            C_value = _hermite_interpolation_particle(C_, C_derivative_, zidx[i] + _W_OFFSET, C_max_idx)
            z[i] = compute_z(sigma_value, tau_value, h_value, C_value, zeta_value)

    return compute_z_kernel_function


def _cubic_vertical_transform_compute_zidx_kernel(
    NZ: int,
    hc: float,
    C: npt.NDArray[np.float64],
    sigma_method: str,
    Vtransform: int,
    horizontal_stencil_half_width: int,
    halley_iterations: int,
    only_initialising: bool,
) -> KernelFunction:
    """Build the compiled kernel function backing `CubicVerticalTransform.compute_zidx_kernel`.

    `C`, the sigma coordinate tabulated from `sigma_method`, and both of their node derivatives
    are captured by closure - `field_data_keys` below is `["h", "zeta"]` only.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels. Baked into the compiled kernel as a compile-time
        constant (via the tabulated `sigma_w` array it determines the shape of).
    hc : float
        The critical depth. Baked into the compiled kernel as a compile-time constant.
    C : npt.NDArray[np.float64]
        `Cs_w`, the complete stretching function array at w points, with exactly `NZ + 1` points.
        Captured by closure.
    sigma_method : str
        Either "linear" or "quadratic"; determines the tabulated sigma array captured by closure.
    Vtransform : int
        Either 1 or 2; determines the `tau(hc, h)` function captured by closure.
    horizontal_stencil_half_width : int
        The half-width `N` of the Lagrange interpolation stencil used for `h` and `zeta`.
    halley_iterations : int
        The number of Halley iterations used to invert `S` on a cubic cell.
    only_initialising : bool
        If False, compute for any active particle. If True, compute only for particles with
        status ``Status.INITIALISING``.

    Returns
    -------
    KernelFunction
        A kernel function that computes the vertical index `zidx`, specialized for the given
        `NZ`, `hc`, `C`, `sigma_method`, `Vtransform`, `horizontal_stencil_half_width`,
        `halley_iterations`, and `only_initialising`.
    """
    hc_: np.float64 = np.float64(hc)
    C_: npt.NDArray[np.float64] = np.array(C, dtype=np.float64)
    C_derivative_: npt.NDArray[np.float64] = _node_derivative_array(C_)
    sigma_w_: npt.NDArray[np.float64] = _sigma_w_array(NZ, sigma_method)
    sigma_w_derivative_: npt.NDArray[np.float64] = _node_derivative_array(sigma_w_)
    tau_fn = _tau_fn_for_Vtransform(Vtransform)
    N = int(horizontal_stencil_half_width)
    horizontal_interpolation_particle = lagrange_2D_particle_factory(N, N)
    compute_zidx_cubic = _compute_zidx_cubic_factory(halley_iterations)
    only_initialising_: bool = bool(only_initialising)

    @kernel_function(
        particle_property_keys=["status", "zidx", "yidx", "xidx", "z"],
        field_data_keys=["h", "zeta"],
    )
    @numba.njit(parallel=True, nogil=True, fastmath=True)
    def compute_zidx_kernel_function(
        status: npt.NDArray[np.uint8],
        zidx: npt.NDArray[np.float64],
        yidx: npt.NDArray[np.float64],
        xidx: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        h_array: npt.NDArray[np.float64],
        h_offy: float,
        h_offx: float,
        zeta_array: npt.NDArray[np.float64],
        zeta_offy: float,
        zeta_offx: float,
    ) -> None:
        """Compute the vertical index 'zidx' of particles in ROMS vertical coordinates.

        Parameters
        ----------
        status : npt.NDArray[np.uint8]
            The particle property array of statuses.
        zidx : npt.NDArray[np.float64]
            The particle property array to store the computed vertical index.
        yidx : npt.NDArray[np.float64]
            The particle property array of y indices.
        xidx : npt.NDArray[np.float64]
            The particle property array of x indices.
        z : npt.NDArray[np.float64]
            The particle property array of physical vertical positions.
        h_array : npt.NDArray[np.float64]
            The bathymetry field array.
        h_offy : float
            The y offset for the bathymetry field.
        h_offx : float
            The x offset for the bathymetry field.
        zeta_array : npt.NDArray[np.float64]
            The free surface field array.
        zeta_offy : float
            The y offset for the free surface field.
        zeta_offx : float
            The x offset for the free surface field.

        Raises
        ------
        ValueError
            If the h or zeta field arrays do not have at least `2 * horizontal_stencil_half_width`
            points in the relevant dimension(s) to avoid out-of-bounds memory access.
        """
        h_max_idx_0 = h_array.shape[0] - 2 * N
        h_max_idx_1 = h_array.shape[1] - 2 * N
        zeta_max_idx_0 = zeta_array.shape[0] - 2 * N
        zeta_max_idx_1 = zeta_array.shape[1] - 2 * N
        if h_max_idx_0 < 0 or h_max_idx_1 < 0:
            raise ValueError(
                "h field must have at least 2 * horizontal_stencil_half_width points in each "
                "dimension to avoid out-of-bounds memory access."
            )
        if zeta_max_idx_0 < 0 or zeta_max_idx_1 < 0:
            raise ValueError(
                "zeta field must have at least 2 * horizontal_stencil_half_width points in each "
                "dimension to avoid out-of-bounds memory access."
            )

        for i in numba.prange(status.shape[0]):  # ty: ignore[not-iterable]
            # only_initialising_ is a compile-time constant, so this branch is optimized away at compile time
            if only_initialising_:
                if status[i] != _INITIALISING:  # only compute for initialising particles
                    continue
            else:
                if status[i] & INACTIVE_FLAG:  # only compute for active particles
                    continue

            h_value = horizontal_interpolation_particle(
                h_array, yidx[i] + h_offy, xidx[i] + h_offx, h_max_idx_0, h_max_idx_1
            )
            zeta_value = horizontal_interpolation_particle(
                zeta_array, yidx[i] + zeta_offy, xidx[i] + zeta_offx, zeta_max_idx_0, zeta_max_idx_1
            )
            tau_value = tau_fn(hc_, h_value)
            zidx[i] = compute_zidx_cubic(
                z[i], h_value, zeta_value, tau_value, sigma_w_, sigma_w_derivative_, C_, C_derivative_
            )

    return compute_zidx_kernel_function
