"""Numba functions for working with ROMS' sigma coordinate scheme.

We follow the same variable naming conventions as ROMS:
- zidx: vertical index
- Nz: total number of vertical rho levels
- hc: critical depth
- sigma: sigma coordinate (uniformly spaced between -1 and 0)
- h: bathymetric depth
- C: stretching function value
- S: S-coordinate value (stretched sigma coordinate)
- zeta: free surface elevation
- z: physical vertical coordinate

Notes
-----
`C` (the vertical stretching function) must be strictly increasing along the vertical axis. This
is a physical requirement of the ROMS S-coordinate system: a valid vertical grid never has
degenerate or crossing levels. `compute_zidx`'s binary search over `C` assumes this; behavior is
undefined if it does not hold.
"""

from dataclasses import dataclass, field

import numba
import numpy as np
import numpy.typing as npt

from .._kernels import KernelFunction, kernel_function
from ..interpolation import lagrange_2D_particle_factory, linear_interpolation_particle
from ..status import INACTIVE_FLAG, Status

_INITIALISING = np.uint8(Status.INITIALISING)

#############################
### computing z from zidx ###
#############################


@numba.njit(nogil=True, fastmath=True)
def _sigma_coordinate(zidx: float, Nz: int) -> float:
    """Compute the sigma coordinate from a vertical index.

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
def compute_z(zidx: float, Nz: int, hc: float, h: float, C: float, zeta: float) -> float:
    """Compute the physical vertical coordinate from the vertical index.

    Parameters
    ----------
    zidx : float
        The vertical index.
    Nz : int
        The total number of vertical rho levels.
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
    sigma = _sigma_coordinate(zidx, Nz)
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
def _sigma_from_Cidx(Cidx: int, C_offset: float, Nz: int) -> float:
    """Compute the sigma coordinate from a stretching function index.

    Parameters
    ----------
    Cidx : int
        The index into the stretching function array.
    C_offset : float
        The offset of the stretching function field relative to the particle's vertical index.
    Nz : int
        The total number of vertical rho levels.

    Returns
    -------
    float
        The sigma coordinate.
    """
    zidx = Cidx - C_offset
    return _sigma_coordinate(zidx, Nz)


@numba.njit(nogil=True, fastmath=True)
def _compute_Cidx_from_S(
    S: float,
    hc: float,
    NZ: int,
    h: float,
    zeta: float,
    C: npt.NDArray[np.float64],
    C_offset: float,
) -> int:
    """Compute the C array index from the S coordinate.

    C_idx corresponds to the index in the stretching function array C such that
        S_coordinate(hc, sigma_from_Cidx(C_idx, C_offset, NZ), h, C[C_idx]) <= S
    and
        S_coordinate(hc, sigma_from_Cidx(C_idx + 1, C_offset, NZ), h, C[C_idx + 1]) > S
    This is done via a binary search over the C array, which assumes C is strictly increasing
    (see the module docstring). If S is outside the range of S values defined by C, the first or
    penultimate index is returned.

    Parameters
    ----------
    S : float
        The S-coordinate value.
    hc : float
        The critical depth.
    NZ : int
        The total number of vertical rho levels.
    h : float
        The bathymetric depth.
    zeta : float
        The free surface elevation. Unused, but kept for a consistent argument bundle with the
        other S-related functions.
    C : npt.NDArray[np.float64]
        The stretching function array. Must be strictly increasing.
    C_offset : float
        The offset of the stretching function field relative to the particle's vertical index.

    Returns
    -------
    int
        The index into C bracketing S, clamped to [0, C.shape[0] - 2].
    """
    C_size = C.shape[0]
    lo = 0
    hi = C_size - 2

    # handle edge cases where S is outside the range of S values
    if S <= _S_coordinate(hc, _sigma_from_Cidx(0, C_offset, NZ), h, C[0]):
        return 0
    elif S >= _S_coordinate(hc, _sigma_from_Cidx(C_size - 2, C_offset, NZ), h, C[C_size - 2]):
        return C_size - 2

    # binary search
    while lo < hi - 1:
        mid = (lo + hi) // 2
        S_mid = _S_coordinate(hc, _sigma_from_Cidx(mid, C_offset, NZ), h, C[mid])
        if S_mid <= S:
            lo = mid
        else:
            hi = mid
    return lo


@numba.njit(nogil=True, fastmath=True)
def _zidx_from_S(
    S: float,
    hc: float,
    NZ: int,
    h: float,
    zeta: float,
    C: npt.NDArray[np.float64],
    C_offset: float,
) -> float:
    """Compute the vertical index from the S coordinate.

    Parameters
    ----------
    S : float
        The S-coordinate value.
    hc : float
        The critical depth.
    NZ : int
        The total number of vertical rho levels.
    h : float
        The bathymetric depth.
    zeta : float
        The free surface elevation. Unused, but kept for a consistent argument bundle with the
        other S-related functions.
    C : npt.NDArray[np.float64]
        The stretching function array. Must be strictly increasing.
    C_offset : float
        The offset of the stretching function field relative to the particle's vertical index.

    Returns
    -------
    float
        The vertical index.
    """
    # find the C index corresponding to S
    Cidx = _compute_Cidx_from_S(S, hc, NZ, h, zeta, C, C_offset)
    # linear interpolation to find the fractional index
    S_low = _S_coordinate(hc, _sigma_from_Cidx(Cidx, C_offset, NZ), h, C[Cidx])
    S_high = _S_coordinate(hc, _sigma_from_Cidx(Cidx + 1, C_offset, NZ), h, C[Cidx + 1])
    f = (S - S_low) / (S_high - S_low)
    return Cidx - C_offset + f


@numba.njit(nogil=True, fastmath=True)
def compute_zidx(
    z: float,
    h: float,
    zeta: float,
    hc: float,
    NZ: int,
    C: npt.NDArray[np.float64],
    C_offset: float,
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
    NZ : int
        The total number of vertical rho levels.
    C : npt.NDArray[np.float64]
        The stretching function array. Must be strictly increasing (see the module docstring);
        behavior is undefined otherwise.
    C_offset : float
        The offset of the stretching function field relative to the particle's vertical index.

    Returns
    -------
    float
        The vertical index.
    """
    S = _S_from_z(z, h, zeta)
    return _zidx_from_S(S, hc, NZ, h, zeta, C, C_offset)


#####################################################
### LinearVerticalTransform2 (ROMS Vtransform=2) ###
#####################################################


@dataclass(frozen=True, eq=False)
class LinearVerticalTransform2:
    r"""ROMS vertical transform 2, with the stretching function interpolated linearly.

    The physical vertical position :math:`z` is given in terms of the sigma coordinate
    :math:`\sigma \in [-1, 0]` by

    .. math::

        z = \zeta + (\zeta + h) S(\sigma)

    where :math:`\zeta` is the free surface elevation and :math:`h` is the bathymetric depth. The
    S-coordinate :math:`S` is

    .. math::

        S(\sigma) = \frac{h_c \sigma + h \, C_s(\sigma)}{h_c + h}

    where :math:`h_c` is the critical depth and :math:`C_s` is the stretching function. `Cs_w`
    gives :math:`C_s` at the `NZ + 1` w-point sigma values
    :math:`\sigma_w(j) = j / NZ - 1,\ j = 0, \ldots, NZ`, matching ROMS' own `Cs_w` output
    variable; a valid stretching function satisfies :math:`C_s(-1) = -1` and :math:`C_s(0) = 0`.


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

    Notes
    -----
    Two instances are equal if they have the same `NZ`, `hc`, `horizontal_stencil_half_width`,
    and `Cs_w`. Instances are not hashable.
    """

    NZ: int
    hc: float
    Cs_w: npt.NDArray[np.float64]
    horizontal_stencil_half_width: int = 1
    _kernel_cache: dict[tuple[str, bool], KernelFunction] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.NZ < 2:
            raise ValueError(f"NZ must be at least 2, got {self.NZ}.")
        if self.hc <= 0:
            raise ValueError(f"hc must be strictly positive, got {self.hc}.")
        if self.horizontal_stencil_half_width < 1:
            raise ValueError(
                f"horizontal_stencil_half_width must be a positive integer, got {self.horizontal_stencil_half_width}."
            )

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
        object.__setattr__(self, "horizontal_stencil_half_width", int(self.horizontal_stencil_half_width))
        object.__setattr__(self, "Cs_w", Cs_w)

    def __eq__(self, other: object) -> bool:
        """Compare `NZ`, `hc`, `horizontal_stencil_half_width`, and every element of `Cs_w`.

        Returns
        -------
        bool
            Whether `other` is a `LinearVerticalTransform2` with the same `NZ`, `hc`,
            `horizontal_stencil_half_width`, and `Cs_w`.
        """
        if not isinstance(other, LinearVerticalTransform2):
            return NotImplemented
        return (
            self.NZ == other.NZ
            and self.hc == other.hc
            and self.horizontal_stencil_half_width == other.horizontal_stencil_half_width
            and np.array_equal(self.Cs_w, other.Cs_w)
        )

    __hash__ = None  # defining __eq__ makes instances unhashable

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
                self.NZ, self.hc, self.Cs_w, self.horizontal_stencil_half_width, only_initialising
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
                self.NZ, self.hc, self.Cs_w, self.horizontal_stencil_half_width, only_initialising
            )
        return self._kernel_cache[key]


def _linear_vertical_transform_2_compute_z_kernel(
    NZ: int, hc: float, C: npt.NDArray[np.float64], horizontal_stencil_half_width: int, only_initialising: bool
) -> KernelFunction:
    """Build the compiled kernel function backing `LinearVerticalTransform2.compute_z_kernel`.

    Unlike `compute_z_kernel_function_factory`, `C` is captured by closure rather than declared as
    field data - `field_data_keys` below is `["h", "zeta"]` only, with no `C`/`C_offset` at all.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels. Baked into the compiled kernel as a compile-time
        constant.
    hc : float
        The critical depth. Baked into the compiled kernel as a compile-time constant.
    C : npt.NDArray[np.float64]
        `Cs_w`, the complete stretching function array at w points, with exactly `NZ + 1` points.
        Captured by closure.
    horizontal_stencil_half_width : int
        The half-width `N` of the Lagrange interpolation stencil used for `h` and `zeta`.
    only_initialising : bool
        If False, compute for any active particle. If True, compute only for particles with
        status ``Status.INITIALISING``.

    Returns
    -------
    KernelFunction
        A kernel function that computes the physical vertical position `z`, specialized for the
        given `NZ`, `hc`, `C`, `horizontal_stencil_half_width`, and `only_initialising`.
    """
    hc_: np.float64 = np.float64(hc)
    NZ_: np.int64 = np.int64(NZ)
    C_: npt.NDArray[np.float64] = np.array(C, dtype=np.float64)
    C_max_idx = C_.shape[0] - 2
    # C is Cs_w (w points): a continuous rho-point zidx maps to local array index zidx + 0.5,
    # since w-point j's sigma value equals rho-point (j - 0.5)'s
    C_offset = 0.5
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
            C_value = linear_interpolation_particle(C_, zidx[i] + C_offset, C_max_idx)
            z[i] = compute_z(zidx[i], NZ_, hc_, h_value, C_value, zeta_value)

    return compute_z_kernel_function


def _linear_vertical_transform_2_compute_zidx_kernel(
    NZ: int, hc: float, C: npt.NDArray[np.float64], horizontal_stencil_half_width: int, only_initialising: bool
) -> KernelFunction:
    """Build the compiled kernel function backing `LinearVerticalTransform2.compute_zidx_kernel`.

    Unlike `compute_zidx_kernel_function_factory`, `C` is captured by closure rather than declared
    as field data - `field_data_keys` below is `["h", "zeta"]` only.

    Parameters
    ----------
    NZ : int
        The total number of vertical rho levels. Baked into the compiled kernel as a compile-time
        constant.
    hc : float
        The critical depth. Baked into the compiled kernel as a compile-time constant.
    C : npt.NDArray[np.float64]
        `Cs_w`, the complete stretching function array at w points, with exactly `NZ + 1` points.
        Captured by closure.
    horizontal_stencil_half_width : int
        The half-width `N` of the Lagrange interpolation stencil used for `h` and `zeta`.
    only_initialising : bool
        If False, compute for any active particle. If True, compute only for particles with
        status ``Status.INITIALISING``.

    Returns
    -------
    KernelFunction
        A kernel function that computes the vertical index `zidx`, specialized for the given
        `NZ`, `hc`, `C`, `horizontal_stencil_half_width`, and `only_initialising`.
    """
    hc_: np.float64 = np.float64(hc)
    NZ_: np.int64 = np.int64(NZ)
    C_: npt.NDArray[np.float64] = np.array(C, dtype=np.float64)
    # C is Cs_w (w points): a continuous rho-point zidx maps to local array index zidx + 0.5,
    # since w-point j's sigma value equals rho-point (j - 0.5)'s - see the class docstring
    C_offset = 0.5
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
            zidx[i] = compute_zidx(z[i], h_value, zeta_value, hc_, NZ_, C_, C_offset)

    return compute_zidx_kernel_function
