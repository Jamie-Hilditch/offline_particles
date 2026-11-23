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

Notes
-----
Both sigma and C (the stretching function) are defined only at the `NZ + 1` w points. `S`, and
therefore `z`, is defined on the continuous vertical index by interpolating between those w-point
values - linearly, for `LinearVerticalTransform2`. `compute_zidx` is an
*exact* inverse of `compute_z` on every grid cell, for either `sigma_method`, because both
directions operate on the same interpolated representation of `S`.
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

# the w-point-to-rho-point-index offset: w-point j's sigma/C value equals rho-point (j - 0.5)'s
_C_OFFSET = 0.5

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
    return np.array([sigma_fn(ki - _C_OFFSET, NZ) for ki in k])


#############################
### computing z from zidx ###
#############################


@numba.njit(nogil=True, fastmath=True)
def _S_coordinate(hc: float, sigma: float, h: float, C: float) -> float:
    """Compute the S-coordinate transformation for ROMS vertical coordinates.

    Parameters
    ----------
    hc : float
        The critical depth.
    sigma : float
        The sigma coordinate.
    h : float
        The bathymetric depth.
    C : float
        The stretching function value.

    Returns
    -------
    float
        The S-coordinate value.
    """
    return (hc * sigma + h * C) / (hc + h)


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
def compute_z(sigma: float, hc: float, h: float, C: float, zeta: float) -> float:
    """Compute the physical vertical coordinate from already-resolved sigma and C values.

    `sigma` and `C` must already be resolved at the particle's vertical index (e.g. by linearly
    interpolating the tabulated `sigma_w`/`Cs_w` arrays) - this function only applies the
    S-coordinate and z-coordinate transforms.

    Parameters
    ----------
    sigma : float
        The sigma coordinate value.
    hc : float
        The critical depth.
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
    S = _S_coordinate(hc, sigma, h, C)
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
def _compute_Cidx_from_S(
    S: float,
    hc: float,
    h: float,
    sigma_w: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
) -> int:
    """Compute the w-point array index bracketing the S coordinate.

    C_idx corresponds to the index such that
        S_coordinate(hc, sigma_w[C_idx], h, C[C_idx]) <= S
    and
        S_coordinate(hc, sigma_w[C_idx + 1], h, C[C_idx + 1]) > S
    This is done via a binary search over `C` (equivalently `sigma_w`, combined through
    `_S_coordinate`), which assumes `C` is strictly increasing (see the module docstring). If S is
    outside the range of S values defined by `C`, the first or penultimate index is returned.

    Parameters
    ----------
    S : float
        The S-coordinate value.
    hc : float
        The critical depth.
    h : float
        The bathymetric depth.
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
    if S <= _S_coordinate(hc, sigma_w[0], h, C[0]):
        return 0
    elif S >= _S_coordinate(hc, sigma_w[C_size - 2], h, C[C_size - 2]):
        return C_size - 2

    # binary search
    while lo < hi - 1:
        mid = (lo + hi) // 2
        S_mid = _S_coordinate(hc, sigma_w[mid], h, C[mid])
        if S_mid <= S:
            lo = mid
        else:
            hi = mid
    return lo


@numba.njit(nogil=True, fastmath=True)
def _zidx_from_S(
    S: float,
    hc: float,
    h: float,
    sigma_w: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
) -> float:
    """Compute the vertical index from the S coordinate.

    Parameters
    ----------
    S : float
        The S-coordinate value.
    hc : float
        The critical depth.
    h : float
        The bathymetric depth.
    sigma_w : npt.NDArray[np.float64]
        The sigma coordinate tabulated at the w points. Must have the same length as `C`.
    C : npt.NDArray[np.float64]
        The stretching function array. Must be strictly increasing.

    Returns
    -------
    float
        The vertical index.
    """
    # find the w-point index bracketing S
    Cidx = _compute_Cidx_from_S(S, hc, h, sigma_w, C)
    # linear interpolation to find the fractional index - exact, since this is the same
    # linear-interpolation-between-w-points representation of S that the forward transform uses
    S_low = _S_coordinate(hc, sigma_w[Cidx], h, C[Cidx])
    S_high = _S_coordinate(hc, sigma_w[Cidx + 1], h, C[Cidx + 1])
    f = (S - S_low) / (S_high - S_low)
    return Cidx - _C_OFFSET + f


@numba.njit(nogil=True, fastmath=True)
def compute_zidx(
    z: float,
    h: float,
    zeta: float,
    hc: float,
    sigma_w: npt.NDArray[np.float64],
    C: npt.NDArray[np.float64],
) -> float:
    """Compute the vertical index from the physical vertical coordinate.

    Parameters
    ----------
    z : float
        The physical vertical coordinate.
    h : float
        The bathymetric depth.
    zeta : float
        The free surface elevation.
    hc : float
        The critical depth.
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
    return _zidx_from_S(S, hc, h, sigma_w, C)


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


#####################################################
### VerticalTransform (abstract base class)      ###
#####################################################


@dataclass(frozen=True, eq=False)
class VerticalTransform(abc.ABC):
    r"""Abstract base class for ROMS vertical coordinate transforms.

    The physical vertical position :math:`z` is given in terms of the sigma coordinate
    :math:`\sigma \in [-1, 0]` by

    .. math::

        z = \zeta + (\zeta + h) S(\sigma)

    where :math:`\zeta` is the free surface elevation and :math:`h` is the bathymetric depth. The
    S-coordinate :math:`S` is

    .. math::

        S(\sigma) = \frac{h_c \sigma + h \, C_s(\sigma)}{h_c + h}

    where :math:`h_c` is the critical depth and :math:`C_s` is the stretching function. Both
    :math:`\sigma` and :math:`C_s` are defined only at the `NZ + 1` w-point sigma values -
    `Cs_w` gives :math:`C_s` directly (matching ROMS' own `Cs_w` output variable), while
    :math:`\sigma` is tabulated internally from `sigma_method`: linearly,
    :math:`\sigma_w(j) = j / NZ - 1` (ROMS Vstretching 1-4), or quadratically (ROMS Vstretching=5,
    see `_sigma_coordinate_quadratic`). A valid stretching function satisfies
    :math:`C_s(-1) = -1` and :math:`C_s(0) = 0`. Concrete subclasses define how `S` (and therefore
    `z`/`zidx`) is computed between these tabulated w points.

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

    Notes
    -----
    Two instances are equal if they are the same concrete type and have equal values for every
    dataclass field (including any added by a subclass). Instances are not hashable.
    """

    NZ: int
    hc: float
    Cs_w: npt.NDArray[np.float64]
    sigma_method: Literal["linear", "quadratic"] = "linear"
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
        """Compare every dataclass field (NZ, hc, Cs_w, sigma_method, and any subclass fields).

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
            Forwarded to the concrete subclass's constructor (e.g. `horizontal_stencil_half_width`).

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
            Forwarded to the concrete subclass's constructor (e.g. `horizontal_stencil_half_width`).

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
            Forwarded to the concrete subclass's constructor (e.g. `horizontal_stencil_half_width`).

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
            Forwarded to the concrete subclass's constructor (e.g. `horizontal_stencil_half_width`).

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
            Forwarded to the concrete subclass's constructor (e.g. `horizontal_stencil_half_width`).

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


#####################################################
### LinearVerticalTransform2 (ROMS Vtransform=2) ###
#####################################################


@dataclass(frozen=True, eq=False)
class LinearVerticalTransform2(VerticalTransform):
    r"""ROMS vertical transform 2, with sigma and the stretching function interpolated linearly.

    `S` (and therefore `z`) is computed by linearly interpolating between the tabulated w-point
    values of `sigma` and `Cs_w` - see `VerticalTransform` for the shared S-coordinate/z-coordinate
    transform and the `NZ`/`hc`/`Cs_w`/`sigma_method` parameters.

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

    Notes
    -----
    Two instances are equal if they have the same `NZ`, `hc`, `Cs_w`, `sigma_method`, and
    `horizontal_stencil_half_width`. Instances are not hashable.
    """

    horizontal_stencil_half_width: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.horizontal_stencil_half_width < 1:
            raise ValueError(
                f"horizontal_stencil_half_width must be a positive integer, got {self.horizontal_stencil_half_width}."
            )
        object.__setattr__(self, "horizontal_stencil_half_width", int(self.horizontal_stencil_half_width))

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
            self._kernel_cache[key] = _linear_vertical_transform_2_compute_z_kernel(
                self.NZ, self.hc, self.Cs_w, self.sigma_method, self.horizontal_stencil_half_width, only_initialising
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
            self._kernel_cache[key] = _linear_vertical_transform_2_compute_zidx_kernel(
                self.NZ, self.hc, self.Cs_w, self.sigma_method, self.horizontal_stencil_half_width, only_initialising
            )
        return self._kernel_cache[key]


def _linear_vertical_transform_2_compute_z_kernel(
    NZ: int,
    hc: float,
    C: npt.NDArray[np.float64],
    sigma_method: str,
    horizontal_stencil_half_width: int,
    only_initialising: bool,
) -> KernelFunction:
    """Build the compiled kernel function backing `LinearVerticalTransform2.compute_z_kernel`.

    `C` and the sigma coordinate tabulated from `sigma_method` are both captured by closure and
    linearly interpolated at each particle's continuous `zidx` - `field_data_keys` below is
    `["h", "zeta"]` only, with no `C`/`sigma`/`C_offset` declared as field data.

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
    horizontal_stencil_half_width : int
        The half-width `N` of the Lagrange interpolation stencil used for `h` and `zeta`.
    only_initialising : bool
        If False, compute for any active particle. If True, compute only for particles with
        status ``Status.INITIALISING``.

    Returns
    -------
    KernelFunction
        A kernel function that computes the physical vertical position `z`, specialized for the
        given `NZ`, `hc`, `C`, `sigma_method`, `horizontal_stencil_half_width`, and
        `only_initialising`.
    """
    hc_: np.float64 = np.float64(hc)
    C_: npt.NDArray[np.float64] = np.array(C, dtype=np.float64)
    C_max_idx = C_.shape[0] - 2
    sigma_w_: npt.NDArray[np.float64] = _sigma_w_array(NZ, sigma_method)
    sigma_max_idx = sigma_w_.shape[0] - 2
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
            sigma_value = linear_interpolation_particle(sigma_w_, zidx[i] + _C_OFFSET, sigma_max_idx)
            C_value = linear_interpolation_particle(C_, zidx[i] + _C_OFFSET, C_max_idx)
            z[i] = compute_z(sigma_value, hc_, h_value, C_value, zeta_value)

    return compute_z_kernel_function


def _linear_vertical_transform_2_compute_zidx_kernel(
    NZ: int,
    hc: float,
    C: npt.NDArray[np.float64],
    sigma_method: str,
    horizontal_stencil_half_width: int,
    only_initialising: bool,
) -> KernelFunction:
    """Build the compiled kernel function backing `LinearVerticalTransform2.compute_zidx_kernel`.

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
    horizontal_stencil_half_width : int
        The half-width `N` of the Lagrange interpolation stencil used for `h` and `zeta`.
    only_initialising : bool
        If False, compute for any active particle. If True, compute only for particles with
        status ``Status.INITIALISING``.

    Returns
    -------
    KernelFunction
        A kernel function that computes the vertical index `zidx`, specialized for the given
        `NZ`, `hc`, `C`, `sigma_method`, `horizontal_stencil_half_width`, and `only_initialising`.
    """
    hc_: np.float64 = np.float64(hc)
    C_: npt.NDArray[np.float64] = np.array(C, dtype=np.float64)
    sigma_w_: npt.NDArray[np.float64] = _sigma_w_array(NZ, sigma_method)
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
            zidx[i] = compute_zidx(z[i], h_value, zeta_value, hc_, sigma_w_, C_)

    return compute_zidx_kernel_function
