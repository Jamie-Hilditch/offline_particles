"""Direct unit tests for the numba ROMS vertical-coordinate helper functions.

Unlike the old Cython `cdef inline` helpers, these numba functions are directly callable from
Python, so they can be unit-tested individually rather than only indirectly through the
black-box kernel tests. Reuses the same hand-derived values already validated against the
reference oracle in `test_reference_oracle.py`, now checking the real implementation.
"""

import numpy as np
import pytest

from offline_particles.kernels.roms.vertical_coordinate._vertical_coordinate import (
    _compute_Cidx_from_S,
    _S_coordinate,
    _S_from_z,
    _sigma_coordinate,
    _z_coordinate,
    _zidx_from_S,
    compute_z,
    compute_zidx,
)

# NZ=4, hc=5.0, h=45.0, zeta=1.0, zidx=1.5:
#   sigma = (1.5 + 0.5) / 4 - 1 = -0.5
#   S = (5*(-0.5) + 45*(-0.5)) / (5 + 45) = -0.5
#   z = 1.0 + (1.0 + 45.0) * (-0.5) = -22.0
_HC = 5.0
_H = 45.0
_ZETA = 1.0
_NZ = 4
_ZIDX = 1.5
_SIGMA = -0.5
_S = -0.5
_Z = -22.0

# C = sigma sampled at each rho level: sigma(k, 4) for k = 0, 1, 2, 3
_LINEAR_C = np.array([-0.875, -0.625, -0.375, -0.125])


def test_sigma_coordinate_matches_hand_derivation() -> None:
    assert _sigma_coordinate(_ZIDX, _NZ) == pytest.approx(_SIGMA)


def test_S_coordinate_matches_hand_derivation() -> None:
    assert _S_coordinate(_HC, _SIGMA, _H, _SIGMA) == pytest.approx(_S)


def test_z_coordinate_matches_hand_derivation() -> None:
    assert _z_coordinate(_S, _H, _ZETA) == pytest.approx(_Z)


def test_compute_z_matches_hand_derivation() -> None:
    assert compute_z(_ZIDX, _NZ, _HC, _H, _SIGMA, _ZETA) == pytest.approx(_Z)


def test_S_from_z_matches_hand_derivation() -> None:
    assert _S_from_z(_Z, _H, _ZETA) == pytest.approx(_S)


def test_compute_Cidx_from_S_finds_the_bracketing_interior_cell() -> None:
    # S=-0.5 lies strictly between S(Cidx=1)=-0.625 and S(Cidx=2)=-0.375 (since C == sigma here,
    # S == sigma exactly at every grid point) so the binary search should land on Cidx=1.
    assert _compute_Cidx_from_S(_S, _HC, _NZ, _H, _ZETA, _LINEAR_C, 0.0) == 1


def test_compute_Cidx_from_S_clamps_below_the_lowest_cell() -> None:
    # S(Cidx=0) == -0.875; anything at or below that clamps to Cidx=0.
    assert _compute_Cidx_from_S(-0.9, _HC, _NZ, _H, _ZETA, _LINEAR_C, 0.0) == 0


def test_compute_Cidx_from_S_clamps_above_the_highest_cell() -> None:
    # S(Cidx=C_size-2=2) == -0.375; anything at or above that clamps to Cidx=C_size-2=2.
    assert _compute_Cidx_from_S(-0.1, _HC, _NZ, _H, _ZETA, _LINEAR_C, 0.0) == 2


def test_zidx_from_S_matches_hand_derivation() -> None:
    assert _zidx_from_S(_S, _HC, _NZ, _H, _ZETA, _LINEAR_C, 0.0) == pytest.approx(_ZIDX)


def test_compute_zidx_matches_hand_derivation() -> None:
    assert compute_zidx(_Z, _H, _ZETA, _HC, _NZ, _LINEAR_C, 0.0) == pytest.approx(_ZIDX)


def test_compute_z_and_compute_zidx_round_trip_on_the_hand_derived_case() -> None:
    z = compute_z(_ZIDX, _NZ, _HC, _H, _sigma_coordinate(_ZIDX, _NZ), _ZETA)
    zidx = compute_zidx(z, _H, _ZETA, _HC, _NZ, _LINEAR_C, 0.0)
    assert zidx == pytest.approx(_ZIDX)
