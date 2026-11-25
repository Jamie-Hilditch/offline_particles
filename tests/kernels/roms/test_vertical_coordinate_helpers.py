"""Direct unit tests for the numba ROMS vertical-coordinate helper functions.

These numba functions are directly callable from Python, so they can be unit-tested individually
rather than only indirectly through the black-box kernel tests. Reuses the same hand-derived
values already validated against the reference oracle in `test_reference_oracle.py`, now checking
the real implementation.
"""

import numpy as np
import pytest

from offline_particles.kernels.roms._vertical_coordinate import (
    _compute_widx_from_S,
    _linear_zidx_from_S,
    _S_coordinate,
    _S_from_z,
    _sigma_coordinate_linear,
    _sigma_coordinate_quadratic,
    _sigma_w_array,
    _tau_Vtransform_1,
    _tau_Vtransform_2,
    _z_coordinate,
    compute_z,
    compute_zidx_linear,
)

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
    assert _sigma_coordinate_linear(_ZIDX, _NZ) == pytest.approx(_SIGMA)


@pytest.mark.parametrize("NZ", [2, 4, 10, 42])
def test_sigma_coordinate_quadratic_boundary_values(NZ: int) -> None:
    assert _sigma_coordinate_quadratic(-0.5, NZ) == pytest.approx(-1.0)
    assert _sigma_coordinate_quadratic(NZ - 0.5, NZ) == pytest.approx(0.0)


def test_sigma_w_array_linear_matches_hand_derivation() -> None:
    sigma_w = _sigma_w_array(_NZ, "linear")
    assert sigma_w == pytest.approx(np.array([-1.0, -0.75, -0.5, -0.25, 0.0]))


@pytest.mark.parametrize("NZ", [2, 4, 10, 42])
def test_sigma_w_array_quadratic_has_exact_boundaries_and_is_strictly_increasing(NZ: int) -> None:
    sigma_w = _sigma_w_array(NZ, "quadratic")
    assert sigma_w[0] == pytest.approx(-1.0)
    assert sigma_w[-1] == pytest.approx(0.0)
    assert np.all(np.diff(sigma_w) > 0)


def test_tau_Vtransform_1_matches_hand_derivation() -> None:
    assert _tau_Vtransform_1(_HC, _H) == pytest.approx(_HC / _H)


def test_tau_Vtransform_2_matches_hand_derivation() -> None:
    assert _tau_Vtransform_2(_HC, _H) == pytest.approx(_TAU)


def test_S_coordinate_matches_hand_derivation() -> None:
    assert _S_coordinate(_TAU, _SIGMA, _SIGMA) == pytest.approx(_S)


def test_z_coordinate_matches_hand_derivation() -> None:
    assert _z_coordinate(_S, _H, _ZETA) == pytest.approx(_Z)


def test_compute_z_matches_hand_derivation() -> None:
    assert compute_z(_SIGMA, _TAU, _H, _SIGMA, _ZETA) == pytest.approx(_Z)


def test_S_from_z_matches_hand_derivation() -> None:
    assert _S_from_z(_Z, _H, _ZETA) == pytest.approx(_S)


def test_compute_widx_from_S_finds_the_bracketing_interior_cell() -> None:
    # S=-0.5 == S(widx=2)==-0.5 exactly (since C == sigma here, S == sigma exactly at every grid
    # point), so the binary search's "S_mid <= S -> lo = mid" rule lands on widx=2.
    assert _compute_widx_from_S(_S, _TAU, _LINEAR_C, _LINEAR_C) == 2


def test_compute_widx_from_S_clamps_below_the_lowest_cell() -> None:
    # S(widx=0) == -1.0; anything at or below that clamps to widx=0.
    assert _compute_widx_from_S(-1.1, _TAU, _LINEAR_C, _LINEAR_C) == 0


def test_compute_widx_from_S_clamps_above_the_highest_cell() -> None:
    # S(widx=C_size-2=3) == -0.25; anything at or above that clamps to widx=C_size-2=3.
    assert _compute_widx_from_S(-0.1, _TAU, _LINEAR_C, _LINEAR_C) == 3


def test_linear_zidx_from_S_matches_hand_derivation() -> None:
    assert _linear_zidx_from_S(_S, _TAU, _LINEAR_C, _LINEAR_C) == pytest.approx(_ZIDX)


def test_compute_zidx_linear_matches_hand_derivation() -> None:
    assert compute_zidx_linear(_Z, _H, _ZETA, _TAU, _LINEAR_C, _LINEAR_C) == pytest.approx(_ZIDX)


def test_compute_z_and_compute_zidx_linear_round_trip_on_the_hand_derived_case() -> None:
    z = compute_z(_sigma_coordinate_linear(_ZIDX, _NZ), _TAU, _H, _SIGMA, _ZETA)
    zidx = compute_zidx_linear(z, _H, _ZETA, _TAU, _LINEAR_C, _LINEAR_C)
    assert zidx == pytest.approx(_ZIDX)
