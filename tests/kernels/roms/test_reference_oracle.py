"""Sanity checks for the reference oracle against hand-derived values.

Every other test file in this package trusts `_reference.py` to produce correct expected
values. If the oracle itself had a bug, it could silently invalidate all of those tests, so
this file checks the oracle against a handful of values worked out by hand directly from the
equations documented in `_vertical_coordinate.py`.
"""

import numpy as np
import pytest

from . import _reference as ref

# NZ=4, hc=5.0, h=45.0, zeta=1.0, zidx=1.5:
#   sigma = (1.5 + 0.5) / 4 - 1 = -0.5
#   tau (Vtransform=2) = 5 / (5 + 45) = 0.1
#   S = 0.1*(-0.5) + 0.9*(-0.5) = -0.5
#   z = 1.0 + (1.0 + 45.0) * (-0.5) = -22.0
_HC = 5.0
_H = 45.0
_ZETA = 1.0
_NZ = 4
_ZIDX = 1.5
_SIGMA = -0.5
_S = -0.5
_Z = -22.0
_TAU = 0.1

# C == the real w-point sigma_w array for NZ=4 (Cs_w == sigma_w exactly, so S == sigma_w exactly
# at every grid point, keeping the hand derivation simple)
_LINEAR_C = np.array([-1.0, -0.75, -0.5, -0.25, 0.0])


def test_sigma_coordinate_linear_matches_hand_derivation() -> None:
    assert ref.sigma_coordinate_linear(_ZIDX, _NZ) == pytest.approx(_SIGMA)


def test_tau_Vtransform_1_matches_hand_derivation() -> None:
    assert ref.tau_Vtransform_1(_HC, _H) == pytest.approx(_HC / _H)


def test_tau_Vtransform_2_matches_hand_derivation() -> None:
    assert ref.tau_Vtransform_2(_HC, _H) == pytest.approx(_TAU)


def test_S_coordinate_matches_hand_derivation() -> None:
    assert ref.S_coordinate(_TAU, _SIGMA, _SIGMA) == pytest.approx(_S)


def test_z_coordinate_matches_hand_derivation() -> None:
    assert ref.z_coordinate(_S, _H, _ZETA) == pytest.approx(_Z)


def test_compute_z_matches_hand_derivation() -> None:
    assert ref.compute_z(_SIGMA, _TAU, _H, _SIGMA, _ZETA) == pytest.approx(_Z)


def test_S_from_z_matches_hand_derivation() -> None:
    assert ref.S_from_z(_Z, _H, _ZETA) == pytest.approx(_S)


def test_compute_widx_from_S_finds_the_bracketing_interior_cell() -> None:
    # S=-0.5 == S(widx=2)==-0.5 exactly (since C == sigma here, S == sigma exactly at every grid
    # point); widx=2 is the correct bracket since S(2) <= S and S < S(3).
    assert ref.compute_widx_from_S(_S, _TAU, _LINEAR_C, _LINEAR_C) == 2


def test_compute_widx_from_S_clamps_below_the_lowest_cell() -> None:
    # S(widx=0) == -1.0; anything at or below that clamps to widx=0.
    assert ref.compute_widx_from_S(-1.1, _TAU, _LINEAR_C, _LINEAR_C) == 0


def test_compute_widx_from_S_clamps_above_the_highest_cell() -> None:
    # S(widx=C_size-2=3) == -0.25; anything at or above that clamps to widx=C_size-2=3.
    assert ref.compute_widx_from_S(-0.1, _TAU, _LINEAR_C, _LINEAR_C) == 3


def test_linear_zidx_from_S_matches_hand_derivation() -> None:
    assert ref.linear_zidx_from_S(_S, _TAU, _LINEAR_C, _LINEAR_C) == pytest.approx(_ZIDX)


def test_compute_zidx_linear_matches_hand_derivation() -> None:
    assert ref.compute_zidx_linear(_Z, _H, _ZETA, _TAU, _LINEAR_C, _LINEAR_C) == pytest.approx(_ZIDX)


def test_node_derivative_array_matches_finite_difference_for_linear_sigma() -> None:
    # sigma_w is tabulated from the linear formula: _bessel_node_derivative_array's interior
    # formula is exact for it everywhere, and the Hyman filter never clamps a constant-slope array.
    sigma_w = ref.sigma_w_array(_NZ, "linear")
    eps = 1e-6
    finite_difference = (
        ref.sigma_coordinate_linear(_ZIDX + eps, _NZ) - ref.sigma_coordinate_linear(_ZIDX - eps, _NZ)
    ) / (2 * eps)
    node_derivative = ref.node_derivative_array(sigma_w)
    # _ZIDX=1.5 -> node 2 (zidx + 0.5 == 2.0), an interior node for NZ=4
    assert node_derivative[2] == pytest.approx(finite_difference)


def test_node_derivative_array_matches_finite_difference_for_quadratic_sigma_away_from_the_dip() -> None:
    # NZ=20 is comfortably above the threshold where the continuous quadratic formula is monotonic
    # between its own tabulated points, so the Hyman filter is a no-op and the Bessel estimate (exact
    # for any degree <=2 polynomial) reproduces the true derivative exactly.
    NZ = 20
    sigma_w = ref.sigma_w_array(NZ, "quadratic")
    zidx = 10.5
    eps = 1e-6
    finite_difference = (
        ref.sigma_coordinate_quadratic(zidx + eps, NZ) - ref.sigma_coordinate_quadratic(zidx - eps, NZ)
    ) / (2 * eps)
    node_derivative = ref.node_derivative_array(sigma_w)
    # zidx=10.5 -> node 11 (zidx + 0.5 == 11.0), an interior node
    assert node_derivative[11] == pytest.approx(finite_difference)


# NZ=4, Cs_w=[-1,-0.9,-0.5,-0.1,0] -> delta=[0.1,0.4,0.4,0.1]. Raw Bessel estimates: interior nodes
# are the arithmetic mean of adjacent secants (0.1+0.4)/2=0.25, (0.4+0.4)/2=0.4, (0.4+0.1)/2=0.25;
# boundary nodes are the one-sided 3-point formula (3*0.1-0.4)/2=-0.05 and (3*0.1-0.4)/2=-0.05
# (symmetric). The Hyman filter: both boundary estimates sign-disagree with their adjacent secant
# (positive) and clamp to 0; the interior caps (3*min(0.1,0.4)=0.3, 3*min(0.4,0.4)=1.2,
# 3*min(0.4,0.1)=0.3) never bind, since 0.25<0.3 and 0.4<1.2.
_CUBIC_CS_W = np.array([-1.0, -0.9, -0.5, -0.1, 0.0])
_CUBIC_CS_W_DERIVATIVE = np.array([0.0, 0.25, 0.4, 0.25, 0.0])


def test_node_derivative_array_matches_hand_derivation_for_cs_w() -> None:
    assert ref.node_derivative_array(_CUBIC_CS_W) == pytest.approx(_CUBIC_CS_W_DERIVATIVE)


def test_hermite_cubic_coefficients_degenerates_to_a_line_for_linear_sigma() -> None:
    # feeding the linear sigma's constant derivative (0.25) at both endpoints of a cell must
    # recover the line itself: a == b == 0.
    a, b, c, d = ref.hermite_cubic_coefficients(-0.5, -0.25, 0.25, 0.25)
    assert a == pytest.approx(0.0, abs=1e-12)
    assert b == pytest.approx(0.0, abs=1e-12)
    assert c == pytest.approx(0.25)
    assert d == pytest.approx(-0.5)


# hand-worked cubic example: NZ=4, hc=5.0, h=45.0, zeta=1.0, Vtransform=2 (tau=0.1), sigma_w
# linear, Cs_w=_CUBIC_CS_W, zidx=1.0 -> cell widx=1, t=0.5:
#   sigma(0.5) == -0.625 (sigma is linear, so the cubic reproduces it exactly)
#   C(0.5): cell-1 Hermite coefficients (f0=-0.9, f1=-0.5, d0=0.25, d1=0.4) are
#     a=2*(-0.9)+0.25-2*(-0.5)+0.4=-0.15, b=-3*(-0.9)-2*0.25+3*(-0.5)-0.4=0.3, c=0.25, d=-0.9
#     -> C(0.5) = ((-0.15*0.5+0.3)*0.5+0.25)*0.5-0.9 = -0.71875
#   S = 0.1*(-0.625) + 0.9*(-0.71875) = -0.709375
#   z = 1.0 + 46.0*(-0.709375) = -31.63125
_CUBIC_ZIDX = 1.0
_CUBIC_Z = -31.63125


def test_cubic_forward_matches_hand_derivation() -> None:
    sigma_w = ref.sigma_w_array(_NZ, "linear")
    sigma_w_derivative = ref.node_derivative_array(sigma_w)
    cs_w_derivative = ref.node_derivative_array(_CUBIC_CS_W)
    sigma_value = ref.hermite_interpolation(sigma_w, sigma_w_derivative, _CUBIC_ZIDX + 0.5)
    C_value = ref.hermite_interpolation(_CUBIC_CS_W, cs_w_derivative, _CUBIC_ZIDX + 0.5)
    assert sigma_value == pytest.approx(-0.625)
    assert C_value == pytest.approx(-0.71875)
    assert ref.compute_z(sigma_value, _TAU, _H, C_value, _ZETA) == pytest.approx(_CUBIC_Z)


def test_compute_zidx_cubic_matches_hand_derivation() -> None:
    # the hand-traced Halley iteration converges from an initial ~4.4% error (seed t0=0.4562,
    # true t=0.5) to a residual around 3e-5 after 1 iteration, ~1e-14 after 2, and ~3e-16 (bit
    # noise) from iteration 3 onward - the default halley_iterations=3 converges far below abs=1e-9
    sigma_w = ref.sigma_w_array(_NZ, "linear")
    sigma_w_derivative = ref.node_derivative_array(sigma_w)
    cs_w_derivative = ref.node_derivative_array(_CUBIC_CS_W)
    zidx = ref.compute_zidx_cubic(
        _CUBIC_Z, _H, _ZETA, _TAU, sigma_w, sigma_w_derivative, _CUBIC_CS_W, cs_w_derivative, halley_iterations=3
    )
    assert zidx == pytest.approx(_CUBIC_ZIDX, abs=1e-9)
