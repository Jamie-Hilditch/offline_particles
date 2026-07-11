"""Property-based round-trip tests for the compiled zidx <-> z transform.

Unlike the other test files, these run entirely against the compiled kernel functions -- forward
then inverse (or vice versa) on the same grid -- so no reference oracle is needed: round-trip
consistency is a property of the real implementation, checked against itself.

As you (the reviewer) pointed out: round-trip invertibility is *not* a blanket guarantee here,
because the inverse has no closed form and is computed via a binary search plus a linear
interpolation in S-space. It happens to be an *exact* inverse of the forward transform on any
given grid cell, because both directions reduce to inverting the same affine map on that cell
(the forward path linearly interpolates C at zidx, and sigma is itself linear in zidx, so z is an
affine function of zidx within a cell; the inverse's binary search locates the correct cell and
then linearly inverts that same affine map). This also extends to mild extrapolation beyond
zidx in [0, NZ-1], because both the forward interpolation's index-clamp and the inverse binary
search's S-value clamp lock onto the *same* boundary cell -- provided the stretching function C
is strictly monotonic with a slope bounded away from zero (so S is never ambiguous) and hc/h/zeta
keep every denominator away from zero. `test_compute_zidx_kernel.py`'s
`test_compute_zidx_swallows_zero_division_on_degenerate_C_segment` documents a case (a flat
C segment) where that precondition fails and round-tripping breaks down -- these hypothesis
strategies deliberately avoid that regime (strictly increasing C, bounded-away-from-zero slope)
rather than re-testing it.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from offline_particles.fields import FieldData
from offline_particles.kernels.roms._vertical_coordinate import (
    compute_z_kernel_function_factory,
    compute_zidx_kernel_function_factory,
)


# a strictly increasing C array with each segment's slope bounded away from zero
@st.composite
def _monotonic_C_array(draw, min_size: int = 4, max_size: int = 10):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    start = draw(st.floats(min_value=-1.0, max_value=-0.5, allow_nan=False, allow_infinity=False))
    increments = draw(
        st.lists(
            st.floats(min_value=0.02, max_value=0.3, allow_nan=False, allow_infinity=False),
            min_size=size - 1,
            max_size=size - 1,
        )
    )
    return np.concatenate([[start], start + np.cumsum(increments)])


# a well-conditioned (hc, h, zeta, C, NZ) grid, plus a zidx within `extrapolation` of [0, NZ-1]
@st.composite
def _roms_grid_and_zidx(draw, extrapolation: float):
    C = draw(_monotonic_C_array())
    NZ = C.shape[0]
    # hc (and NZ, via C's size) select which compiled kernel variant is exercised -- each distinct
    # (hc, NZ) pair triggers a fresh numba compile the first time it's seen (see
    # compute_z_kernel_function_factory/compute_zidx_kernel_function_factory), so hc is drawn from
    # a small fixed set rather than a continuous range to keep the compile cache bounded across
    # examples, instead of recompiling for nearly every one of the (up to 200) draws.
    hc = draw(st.sampled_from([0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0]))
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
    return hc, h, zeta, C, NZ, zidx


def _build_inputs(hc: float, h: float, zeta: float, C: np.ndarray, NZ: int, zidx: float):
    particle_properties = {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([zidx]),
        "yidx": np.array([1.5]),
        "xidx": np.array([1.5]),
        "z": np.array([np.nan]),
    }
    field_data = {
        "h": FieldData(np.full((4, 4), h), (0.0, 0.0)),
        "zeta": FieldData(np.full((4, 4), zeta), (0.0, 0.0)),
        "C": FieldData(C, (0.0,)),
    }
    return particle_properties, field_data


def _round_trip_zidx_to_z_to_zidx(hc: float, h: float, zeta: float, C: np.ndarray, NZ: int, zidx: float):
    particle_properties, field_data = _build_inputs(hc, h, zeta, C, NZ, zidx)
    compute_z_kernel_function = compute_z_kernel_function_factory(hc, NZ)
    compute_zidx_kernel_function = compute_zidx_kernel_function_factory(hc, NZ)

    compute_z_kernel_function(particle_properties, {}, field_data)
    z = particle_properties["z"][0]

    particle_properties["zidx"][0] = np.nan
    compute_zidx_kernel_function(particle_properties, {}, field_data)
    recovered_zidx = particle_properties["zidx"][0]

    return z, recovered_zidx


@settings(max_examples=200, deadline=None)
@given(params=_roms_grid_and_zidx(extrapolation=0.0))
def test_interior_round_trip_zidx_to_z_to_zidx_is_exact(params) -> None:
    hc, h, zeta, C, NZ, zidx = params
    _, recovered_zidx = _round_trip_zidx_to_z_to_zidx(hc, h, zeta, C, NZ, zidx)
    assert recovered_zidx == pytest.approx(zidx, rel=1e-9, abs=1e-9)


@settings(max_examples=200, deadline=None)
@given(params=_roms_grid_and_zidx(extrapolation=1.0))
def test_mild_extrapolation_round_trip_zidx_to_z_to_zidx_is_exact(params) -> None:
    """Extends the interior round-trip property up to one index unit beyond [0, NZ-1].

    See the module docstring for why this still holds for well-conditioned (strictly monotonic,
    bounded-slope) C: the forward interpolation's index-clamp and the inverse binary search's
    boundary clamp both lock onto the same edge cell.
    """
    hc, h, zeta, C, NZ, zidx = params
    _, recovered_zidx = _round_trip_zidx_to_z_to_zidx(hc, h, zeta, C, NZ, zidx)
    assert recovered_zidx == pytest.approx(zidx, rel=1e-9, abs=1e-9)


@settings(max_examples=200, deadline=None)
@given(params=_roms_grid_and_zidx(extrapolation=0.0))
def test_interior_round_trip_z_to_zidx_to_z_is_exact(params) -> None:
    """The mirrored property: re-deriving z from the recovered zidx reproduces the original z."""
    hc, h, zeta, C, NZ, zidx = params
    z, recovered_zidx = _round_trip_zidx_to_z_to_zidx(hc, h, zeta, C, NZ, zidx)

    particle_properties, field_data = _build_inputs(hc, h, zeta, C, NZ, recovered_zidx)
    compute_z_kernel_function_factory(hc, NZ)(particle_properties, {}, field_data)
    recovered_z = particle_properties["z"][0]

    assert recovered_z == pytest.approx(z, rel=1e-9, abs=1e-9)
