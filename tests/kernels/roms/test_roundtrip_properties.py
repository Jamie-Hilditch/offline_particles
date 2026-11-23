"""Property-based round-trip tests for the compiled zidx <-> z transform.

Unlike the other test files, these run entirely against the compiled kernel functions -- forward
then inverse (or vice versa) on the same grid -- so no reference oracle is needed: round-trip
consistency is a property of the real implementation, checked against itself.

As you (the reviewer) pointed out: round-trip invertibility is *not* a blanket guarantee here,
because the inverse has no closed form and is computed via a binary search plus a linear
interpolation in S-space. It happens to be an *exact* inverse of the forward transform on any
given grid cell, because both directions reduce to inverting the same affine map on that cell: both
`sigma` and `Cs_w` are only ever defined/interpolated at the w points (never evaluated from a
continuous nonlinear formula at an arbitrary `zidx`), so `S` - and therefore `z` - is an affine
function of `zidx` within a cell regardless of `sigma_method` or `Vtransform`; the inverse's binary
search locates the correct cell and then linearly inverts that same affine map. This is why neither
`sigma_method` nor `Vtransform` make any difference to the tolerance used below - every combination
is exercised at the same tight tolerance. This also extends to mild extrapolation beyond `zidx` in
`[0, NZ-1]`, because both the forward interpolation's index-clamp and the inverse binary search's
S-value clamp lock onto the *same* boundary cell -- provided the stretching function `Cs_w` is
strictly monotonic with a slope bounded away from zero (so S is never ambiguous) and hc/h/zeta keep
every denominator away from zero -- a flat/degenerate `Cs_w` segment violates this and is not a
valid input (`Cs_w` must be strictly increasing, per `LinearVerticalTransform`'s contract). For
`Vtransform=1` specifically, `hc <= h` must also hold (see `LinearVerticalTransform`'s docstring) -
the `h` strategy below enforces this whenever `Vtransform=1` is drawn.

Unlike the old `@cache`'d `hc`/`NZ`-only factories this replaced, `Cs_w` is baked into the
compiled kernel via closure, and `LinearVerticalTransform` caches compiled kernels per-instance
only, not across instances (see its class docstring) - so a *fresh* `LinearVerticalTransform`
per example would trigger a fresh numba compile every time. `h`, `zeta`, and `zidx`, by contrast,
are ordinary runtime arguments (field data / particle properties), not baked into anything, so
they're free to vary continuously without any compile cost. The strategies below exploit this
split: `Cs_w` is drawn from a small fixed set of candidates (`_CS_W_CANDIDATES`, varying in both
size and curvature) rather than generated fresh per example, and `_get_transform` caches one
`LinearVerticalTransform` instance per `(hc, Cs_w, sigma_method, Vtransform)` combination actually
seen - bounding the total number of compiles accordingly, however many examples run.
"""

from typing import Literal

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from offline_particles.fields import FieldData
from offline_particles.kernels.roms import LinearVerticalTransform

_SigmaMethod = Literal["linear", "quadratic"]
_Vtransform = Literal[1, 2]


def _make_cs_w(size: int, curvature: float) -> np.ndarray:
    # curvature in (-1, 1): 0 is linear-in-sigma (Cs_w == sigma_w); nonzero adds a controlled
    # nonlinear warp while keeping the slope strictly bounded away from zero and exact boundary
    # values Cs_w[0] = -1, Cs_w[-1] = 0 (the warp term vanishes at both endpoints)
    sigma_w = np.linspace(-1.0, 0.0, size)
    return sigma_w + curvature * sigma_w * (sigma_w + 1.0)


# a small, fixed set of strictly-increasing, bounded-slope Cs_w arrays spanning a range of sizes
# and curvatures - see the module docstring for why this is fixed rather than freshly generated
_CS_W_CANDIDATES = [_make_cs_w(size, curvature) for size in (4, 6, 8, 10) for curvature in (-0.3, 0.0, 0.3)]
_HC_CANDIDATES = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0]

_SIGMA_METHOD_CANDIDATES: tuple[_SigmaMethod, ...] = ("linear", "quadratic")
_VTRANSFORM_CANDIDATES: tuple[_Vtransform, ...] = (1, 2)

_transform_cache: dict[tuple[float, int, str, int], LinearVerticalTransform] = {}


def _get_transform(
    hc: float, Cs_w: np.ndarray, sigma_method: _SigmaMethod, Vtransform: _Vtransform
) -> LinearVerticalTransform:
    # Cs_w is always one of the fixed _CS_W_CANDIDATES objects (via st.sampled_from, which yields
    # the original object, not a copy), so id(Cs_w) is a stable, reusable cache key
    key = (hc, id(Cs_w), sigma_method, Vtransform)
    if key not in _transform_cache:
        _transform_cache[key] = LinearVerticalTransform(
            NZ=Cs_w.shape[0] - 1, hc=hc, Cs_w=Cs_w, sigma_method=sigma_method, Vtransform=Vtransform
        )
    return _transform_cache[key]


# a well-conditioned (hc, h, zeta, Cs_w, sigma_method, Vtransform, NZ) grid, plus a zidx within
# `extrapolation` of [0, NZ-1]
@st.composite
def _roms_grid_and_zidx(draw, extrapolation: float):
    Cs_w = draw(st.sampled_from(_CS_W_CANDIDATES))
    NZ = Cs_w.shape[0] - 1
    hc = draw(st.sampled_from(_HC_CANDIDATES))
    sigma_method = draw(st.sampled_from(_SIGMA_METHOD_CANDIDATES))
    Vtransform = draw(st.sampled_from(_VTRANSFORM_CANDIDATES))
    # Vtransform=1 requires hc <= h (see LinearVerticalTransform's docstring); Vtransform=2 has
    # no such constraint
    h_min = hc if Vtransform == 1 else 1.0
    h = draw(st.floats(min_value=h_min, max_value=500.0, allow_nan=False, allow_infinity=False))
    # keep |zeta| well below h so zeta + h stays safely away from zero
    zeta = h * draw(st.floats(min_value=-0.4, max_value=0.4, allow_nan=False, allow_infinity=False))
    zidx = draw(
        st.floats(
            min_value=-extrapolation,
            max_value=(NZ - 1) + extrapolation,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return hc, h, zeta, Cs_w, sigma_method, Vtransform, NZ, zidx


def _build_particle_properties(zidx: float) -> dict[str, np.ndarray]:
    return {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([zidx]),
        "yidx": np.array([1.5]),
        "xidx": np.array([1.5]),
        "z": np.array([np.nan]),
    }


def _round_trip_zidx_to_z_to_zidx(
    hc: float,
    h: float,
    zeta: float,
    Cs_w: np.ndarray,
    sigma_method: _SigmaMethod,
    Vtransform: _Vtransform,
    zidx: float,
):
    transform = _get_transform(hc, Cs_w, sigma_method, Vtransform)
    field_data = {
        "h": FieldData(np.full((4, 4), h), (0.0, 0.0)),
        "zeta": FieldData(np.full((4, 4), zeta), (0.0, 0.0)),
    }

    particle_properties = _build_particle_properties(zidx)
    transform.compute_z_kernel()(particle_properties, {}, field_data)
    z = particle_properties["z"][0]

    particle_properties["zidx"][0] = np.nan
    transform.compute_zidx_kernel()(particle_properties, {}, field_data)
    recovered_zidx = particle_properties["zidx"][0]

    return transform, z, recovered_zidx


@settings(max_examples=200, deadline=None)
@given(params=_roms_grid_and_zidx(extrapolation=0.0))
def test_interior_round_trip_zidx_to_z_to_zidx_is_exact(params) -> None:
    hc, h, zeta, Cs_w, sigma_method, Vtransform, _, zidx = params
    _, _, recovered_zidx = _round_trip_zidx_to_z_to_zidx(hc, h, zeta, Cs_w, sigma_method, Vtransform, zidx)
    assert recovered_zidx == pytest.approx(zidx, rel=1e-9, abs=1e-9)


@settings(max_examples=200, deadline=None)
@given(params=_roms_grid_and_zidx(extrapolation=1.0))
def test_mild_extrapolation_round_trip_zidx_to_z_to_zidx_is_exact(params) -> None:
    """Extends the interior round-trip property up to one index unit beyond [0, NZ-1].

    See the module docstring for why this still holds for well-conditioned (strictly monotonic,
    bounded-slope) `Cs_w`: the forward interpolation's index-clamp and the inverse binary search's
    boundary clamp both lock onto the same edge cell.
    """
    hc, h, zeta, Cs_w, sigma_method, Vtransform, _, zidx = params
    _, _, recovered_zidx = _round_trip_zidx_to_z_to_zidx(hc, h, zeta, Cs_w, sigma_method, Vtransform, zidx)
    assert recovered_zidx == pytest.approx(zidx, rel=1e-9, abs=1e-9)


@settings(max_examples=200, deadline=None)
@given(params=_roms_grid_and_zidx(extrapolation=0.0))
def test_interior_round_trip_z_to_zidx_to_z_is_exact(params) -> None:
    """The mirrored property: re-deriving z from the recovered zidx reproduces the original z."""
    hc, h, zeta, Cs_w, sigma_method, Vtransform, _, zidx = params
    transform, z, recovered_zidx = _round_trip_zidx_to_z_to_zidx(hc, h, zeta, Cs_w, sigma_method, Vtransform, zidx)

    field_data = {
        "h": FieldData(np.full((4, 4), h), (0.0, 0.0)),
        "zeta": FieldData(np.full((4, 4), zeta), (0.0, 0.0)),
    }
    particle_properties = _build_particle_properties(recovered_zidx)
    transform.compute_z_kernel()(particle_properties, {}, field_data)
    recovered_z = particle_properties["z"][0]

    assert recovered_z == pytest.approx(z, rel=1e-9, abs=1e-9)
