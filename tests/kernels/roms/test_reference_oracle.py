"""Sanity checks for the reference oracle against hand-derived values.

Every other test file in this package trusts `_reference.py` to produce correct expected
values. If the oracle itself had a bug, it could silently invalidate all of those tests, so
this file checks the oracle against a handful of values worked out by hand directly from the
equations documented in `_vertical_coordinate.pxd`.
"""

import numpy as np
import pytest

from . import _reference as ref

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
    assert ref.sigma_coordinate(_ZIDX, _NZ) == pytest.approx(_SIGMA)


def test_S_coordinate_matches_hand_derivation() -> None:
    assert ref.S_coordinate(_HC, _SIGMA, _H, _SIGMA) == pytest.approx(_S)


def test_z_coordinate_matches_hand_derivation() -> None:
    assert ref.z_coordinate(_S, _H, _ZETA) == pytest.approx(_Z)


def test_compute_z_matches_hand_derivation() -> None:
    assert ref.compute_z(_ZIDX, _NZ, _HC, _H, _SIGMA, _ZETA) == pytest.approx(_Z)


def test_S_from_z_matches_hand_derivation() -> None:
    assert ref.S_from_z(_Z, _H, _ZETA) == pytest.approx(_S)


def test_compute_Cidx_from_S_finds_the_bracketing_interior_cell() -> None:
    # S=-0.5 lies strictly between S(Cidx=1)=-0.625 and S(Cidx=2)=-0.375 (since C == sigma here,
    # S == sigma exactly at every grid point) so the binary search should land on Cidx=1.
    assert ref.compute_Cidx_from_S(_S, _HC, _NZ, _H, _ZETA, _LINEAR_C, 0.0) == 1


def test_compute_Cidx_from_S_clamps_below_the_lowest_cell() -> None:
    # S(Cidx=0) == -0.875; anything at or below that clamps to Cidx=0.
    assert ref.compute_Cidx_from_S(-0.9, _HC, _NZ, _H, _ZETA, _LINEAR_C, 0.0) == 0


def test_compute_Cidx_from_S_clamps_above_the_highest_cell() -> None:
    # S(Cidx=C_size-2=2) == -0.375; anything at or above that clamps to Cidx=C_size-2=2.
    assert ref.compute_Cidx_from_S(-0.1, _HC, _NZ, _H, _ZETA, _LINEAR_C, 0.0) == 2


def test_zidx_from_S_matches_hand_derivation() -> None:
    assert ref.zidx_from_S(_S, _HC, _NZ, _H, _ZETA, _LINEAR_C, 0.0) == pytest.approx(_ZIDX)


def test_compute_zidx_matches_hand_derivation() -> None:
    assert ref.compute_zidx(_Z, _H, _ZETA, _HC, _NZ, _LINEAR_C, 0.0) == pytest.approx(_ZIDX)


def test_reference_kernel_functions_match_scalar_helpers() -> None:
    # cross-check the whole-kernel reference wrappers against the scalar helpers above, since the
    # kernel tests exercise the wrappers (not the scalar helpers) directly.
    from offline_particles.fields import FieldData

    particle_properties = {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([_ZIDX]),
        "yidx": np.array([1.5]),
        "xidx": np.array([1.5]),
        "z": np.array([np.nan]),
    }
    scalars = {"hc": _HC, "NZ": _NZ}
    field_data = {
        "h": FieldData(np.full((4, 4), _H), (0.0, 0.0)),
        "zeta": FieldData(np.full((4, 4), _ZETA), (0.0, 0.0)),
        "C": FieldData(_LINEAR_C, (0.0,)),
    }

    ref.reference_compute_z_kernel_function(particle_properties, scalars, field_data)
    assert particle_properties["z"][0] == pytest.approx(_Z)

    particle_properties["zidx"] = np.array([np.nan])
    ref.reference_compute_zidx_kernel_function(particle_properties, scalars, field_data)
    assert particle_properties["zidx"][0] == pytest.approx(_ZIDX)
