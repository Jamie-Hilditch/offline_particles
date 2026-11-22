"""Property-based round-trip tests for the compiled zidx <-> z transform.

Unlike the other test files, these run entirely against the compiled kernel functions -- forward
then inverse (or vice versa) on the same grid -- so no reference oracle is needed: round-trip
consistency is a property of the real implementation, checked against itself.

As you (the reviewer) pointed out: round-trip invertibility is *not* a blanket guarantee here,
because the inverse has no closed form and is computed via a binary search plus a linear
interpolation in S-space. It happens to be an *exact* inverse of the forward transform on any
given grid cell, because both directions reduce to inverting the same affine map on that cell
(the forward path linearly interpolates `Cs_w` at `zidx`, and sigma is itself linear in `zidx`, so
`z` is an affine function of `zidx` within a cell; the inverse's binary search locates the correct
cell and then linearly inverts that same affine map). This also extends to mild extrapolation
beyond `zidx` in `[0, NZ-1]`, because both the forward interpolation's index-clamp and the inverse
binary search's S-value clamp lock onto the *same* boundary cell -- provided the stretching
function `Cs_w` is strictly monotonic with a slope bounded away from zero (so S is never
ambiguous) and hc/h/zeta keep every denominator away from zero -- a flat/degenerate `Cs_w` segment
violates this and is not a valid input (`Cs_w` must be strictly increasing, per
`LinearVerticalTransform2`'s contract).

Unlike the old `@cache`'d `hc`/`NZ`-only factories this replaced, `Cs_w` is baked into the
compiled kernel via closure, and `LinearVerticalTransform2` caches compiled kernels per-instance
only, not across instances (see its class docstring) - so a *fresh* `LinearVerticalTransform2`
per example would trigger a fresh numba compile every time. `h`, `zeta`, and `zidx`, by contrast,
are ordinary runtime arguments (field data / particle properties), not baked into anything, so
they're free to vary continuously without any compile cost. The strategies below exploit this
split: `Cs_w` is drawn from a small fixed set of candidates (`_CS_W_CANDIDATES`, varying in both
size and curvature) rather than generated fresh per example, and `_get_transform` caches one
`LinearVerticalTransform2` instance per `(hc, Cs_w)` pair actually seen - bounding the total number
of compiles to (at most) `len(_HC_CANDIDATES) * len(_CS_W_CANDIDATES)`, however many examples run.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from offline_particles.fields import FieldData
from offline_particles.kernels.roms import LinearVerticalTransform2


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

_transform_cache: dict[tuple[float, int], LinearVerticalTransform2] = {}


def _get_transform(hc: float, Cs_w: np.ndarray) -> LinearVerticalTransform2:
    # Cs_w is always one of the fixed _CS_W_CANDIDATES objects (via st.sampled_from, which yields
    # the original object, not a copy), so id(Cs_w) is a stable, reusable cache key
    key = (hc, id(Cs_w))
    if key not in _transform_cache:
        _transform_cache[key] = LinearVerticalTransform2(NZ=Cs_w.shape[0] - 1, hc=hc, Cs_w=Cs_w)
    return _transform_cache[key]


# a well-conditioned (hc, h, zeta, Cs_w, NZ) grid, plus a zidx within `extrapolation` of [0, NZ-1]
@st.composite
def _roms_grid_and_zidx(draw, extrapolation: float):
    Cs_w = draw(st.sampled_from(_CS_W_CANDIDATES))
    NZ = Cs_w.shape[0] - 1
    hc = draw(st.sampled_from(_HC_CANDIDATES))
    h = draw(st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False))
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
    return hc, h, zeta, Cs_w, NZ, zidx


def _build_particle_properties(zidx: float) -> dict[str, np.ndarray]:
    return {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([zidx]),
        "yidx": np.array([1.5]),
        "xidx": np.array([1.5]),
        "z": np.array([np.nan]),
    }


def _round_trip_zidx_to_z_to_zidx(hc: float, h: float, zeta: float, Cs_w: np.ndarray, zidx: float):
    transform = _get_transform(hc, Cs_w)
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
    hc, h, zeta, Cs_w, _, zidx = params
    _, _, recovered_zidx = _round_trip_zidx_to_z_to_zidx(hc, h, zeta, Cs_w, zidx)
    assert recovered_zidx == pytest.approx(zidx, rel=1e-9, abs=1e-9)


@settings(max_examples=200, deadline=None)
@given(params=_roms_grid_and_zidx(extrapolation=1.0))
def test_mild_extrapolation_round_trip_zidx_to_z_to_zidx_is_exact(params) -> None:
    """Extends the interior round-trip property up to one index unit beyond [0, NZ-1].

    See the module docstring for why this still holds for well-conditioned (strictly monotonic,
    bounded-slope) `Cs_w`: the forward interpolation's index-clamp and the inverse binary search's
    boundary clamp both lock onto the same edge cell.
    """
    hc, h, zeta, Cs_w, _, zidx = params
    _, _, recovered_zidx = _round_trip_zidx_to_z_to_zidx(hc, h, zeta, Cs_w, zidx)
    assert recovered_zidx == pytest.approx(zidx, rel=1e-9, abs=1e-9)


@settings(max_examples=200, deadline=None)
@given(params=_roms_grid_and_zidx(extrapolation=0.0))
def test_interior_round_trip_z_to_zidx_to_z_is_exact(params) -> None:
    """The mirrored property: re-deriving z from the recovered zidx reproduces the original z."""
    hc, h, zeta, Cs_w, _, zidx = params
    transform, z, recovered_zidx = _round_trip_zidx_to_z_to_zidx(hc, h, zeta, Cs_w, zidx)

    field_data = {
        "h": FieldData(np.full((4, 4), h), (0.0, 0.0)),
        "zeta": FieldData(np.full((4, 4), zeta), (0.0, 0.0)),
    }
    particle_properties = _build_particle_properties(recovered_zidx)
    transform.compute_z_kernel()(particle_properties, {}, field_data)
    recovered_z = particle_properties["z"][0]

    assert recovered_z == pytest.approx(z, rel=1e-9, abs=1e-9)
