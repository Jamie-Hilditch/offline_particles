"""Tests for the ROMS Vstretching 1-5 helper functions and `from_Vstretching_1..5` classmethods.

Each `_Cs_w_Vstretching_N` is checked against an independent, from-scratch re-implementation of
the formula given in ROMS's `set_scoord.F` (not against the production code's own internal
building blocks), so a transcription bug in the production formula would not also appear - by
construction - in these reference values.
"""

import numpy as np
import pytest

from offline_particles.kernels.roms._vertical_coordinate import (
    LinearVerticalTransform2,
    VerticalTransform,
    _C_double_stretching,
    _Cs_w_Vstretching_1,
    _Cs_w_Vstretching_2,
    _Cs_w_Vstretching_3,
    _Cs_w_Vstretching_4,
    _Cs_w_Vstretching_5,
    _sigma_w_array,
)

_NZ = 8
# Vstretching=3 is deliberately excluded from this shared list: unlike 1/2/4/5, both of its
# parameters (alpha, beta) must stay strictly positive (ROMS documents alpha in [0.65, 1.0],
# beta in [0.58, 3.0] - no zero value is valid), so it gets its own sweep below.
_HELPERS_WITH_ZERO_VALID = [_Cs_w_Vstretching_1, _Cs_w_Vstretching_2, _Cs_w_Vstretching_4, _Cs_w_Vstretching_5]
_FROM_VSTRETCHING_NAMES = [
    "from_Vstretching_1",
    "from_Vstretching_2",
    "from_Vstretching_3",
    "from_Vstretching_4",
    "from_Vstretching_5",
]


# --- boundary conditions and monotonicity, swept over valid (theta_s, theta_b) ---


@pytest.mark.parametrize("helper", _HELPERS_WITH_ZERO_VALID)
@pytest.mark.parametrize("theta_s", [0.1, 1.0, 5.0, 8.0])
@pytest.mark.parametrize("theta_b", [0.0, 0.5, 1.0])
def test_boundary_conditions_and_strict_monotonicity(helper, theta_s: float, theta_b: float) -> None:
    Cs_w = helper(_NZ, theta_s, theta_b)

    assert Cs_w.shape == (_NZ + 1,)
    assert Cs_w[0] == pytest.approx(-1.0)
    assert Cs_w[-1] == pytest.approx(0.0)
    assert np.all(np.diff(Cs_w) > 0)


@pytest.mark.parametrize("alpha", [0.65, 1.0])
@pytest.mark.parametrize("beta", [0.58, 1.0, 3.0])
def test_Vstretching_3_boundary_conditions_and_strict_monotonicity(alpha: float, beta: float) -> None:
    Cs_w = _Cs_w_Vstretching_3(_NZ, alpha, beta)

    assert Cs_w.shape == (_NZ + 1,)
    assert Cs_w[0] == pytest.approx(-1.0)
    assert Cs_w[-1] == pytest.approx(0.0)
    assert np.all(np.diff(Cs_w) > 0)


# --- degenerate branches reduce to their documented simplified forms ---


def test_Vstretching_1_theta_s_zero_reduces_to_sigma() -> None:
    Cs_w = _Cs_w_Vstretching_1(_NZ, theta_s=0.0, theta_b=0.3)
    assert Cs_w == pytest.approx(_sigma_w_array(_NZ, "linear"))


def test_Vstretching_2_theta_s_zero_and_theta_b_zero_reduces_to_sigma() -> None:
    Cs_w = _Cs_w_Vstretching_2(_NZ, theta_s=0.0, theta_b=0.0)
    assert Cs_w == pytest.approx(_sigma_w_array(_NZ, "linear"))


def test_Vstretching_2_theta_b_zero_equals_Csur_only() -> None:
    sigma_w = _sigma_w_array(_NZ, "linear")
    Csur = (1 - np.cosh(2.5 * sigma_w)) / (np.cosh(2.5) - 1)
    Cs_w = _Cs_w_Vstretching_2(_NZ, theta_s=2.5, theta_b=0.0)
    assert Cs_w == pytest.approx(Csur)


def test_Vstretching_4_theta_s_zero_uses_minus_sigma_squared() -> None:
    sigma_w = _sigma_w_array(_NZ, "linear")
    Csur = -(sigma_w**2)
    expected = (np.exp(1.5 * Csur) - 1) / (1 - np.exp(-1.5))
    Cs_w = _Cs_w_Vstretching_4(_NZ, theta_s=0.0, theta_b=1.5)
    assert Cs_w == pytest.approx(expected)


def test_Vstretching_4_theta_b_zero_equals_Csur_only() -> None:
    sigma_w = _sigma_w_array(_NZ, "linear")
    Csur = (1 - np.cosh(4.0 * sigma_w)) / (np.cosh(4.0) - 1)
    Cs_w = _Cs_w_Vstretching_4(_NZ, theta_s=4.0, theta_b=0.0)
    assert Cs_w == pytest.approx(Csur)


# --- independent from-scratch reference implementations of the ROMS set_scoord.F formulae ---


def _ref_sigma_w_linear(NZ: int) -> np.ndarray:
    k = np.arange(NZ + 1, dtype=np.float64)
    return k / NZ - 1.0


def _ref_Vstretching_1(NZ: int, theta_s: float, theta_b: float) -> np.ndarray:
    s = _ref_sigma_w_linear(NZ)
    if theta_s == 0:
        return s
    return (1 - theta_b) * np.sinh(theta_s * s) / np.sinh(theta_s) + theta_b * (
        -0.5 + 0.5 * np.tanh(theta_s * (s + 0.5)) / np.tanh(0.5 * theta_s)
    )


def _ref_Vstretching_2(NZ: int, theta_s: float, theta_b: float) -> np.ndarray:
    s = _ref_sigma_w_linear(NZ)
    Csur = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1) if theta_s > 0 else s
    if theta_b > 0:
        Cbot = np.sinh(theta_b * (s + 1)) / np.sinh(theta_b) - 1
        Aweight = Bweight = 1.0
        Cweight = (s + 1) ** Aweight * (1 + (Aweight / Bweight) * (1 - (s + 1) ** Bweight))
        return Cweight * Csur + (1 - Cweight) * Cbot
    return Csur


def _ref_Vstretching_3(NZ: int, theta_s: float, theta_b: float) -> np.ndarray:
    s = _ref_sigma_w_linear(NZ)
    H = 3.0
    alpha, beta = theta_s, theta_b
    Csur = -np.log(np.cosh(H * np.abs(s) ** alpha)) / np.log(np.cosh(H))
    Cbot = np.log(np.cosh(H * (s + 1) ** beta)) / np.log(np.cosh(H)) - 1
    Cweight = 0.5 * (1 - np.tanh(H * (s + 0.5)))
    return Cweight * Cbot + (1 - Cweight) * Csur


def _ref_double_stretching(s: np.ndarray, theta_s: float, theta_b: float) -> np.ndarray:
    Csur = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1) if theta_s > 0 else -(s**2)
    if theta_b > 0:
        return (np.exp(theta_b * Csur) - 1) / (1 - np.exp(-theta_b))
    return Csur


def _ref_sigma_w_quadratic(NZ: int) -> np.ndarray:
    k = np.arange(NZ + 1, dtype=np.float64)
    N = float(NZ)
    return -(k**2 - 2 * k * N + k + N**2 - N) / (N**2 - N) - 0.01 * (k**2 - k * N) / (1.0 - N)


@pytest.mark.parametrize("theta_s", [0.5, 2.0, 7.5])
@pytest.mark.parametrize("theta_b", [0.0, 0.3, 1.0])
def test_Vstretching_1_matches_independent_reference(theta_s: float, theta_b: float) -> None:
    assert _Cs_w_Vstretching_1(_NZ, theta_s, theta_b) == pytest.approx(_ref_Vstretching_1(_NZ, theta_s, theta_b))


@pytest.mark.parametrize("theta_s", [0.5, 2.0, 7.5])
@pytest.mark.parametrize("theta_b", [0.0, 0.3, 1.0])
def test_Vstretching_2_matches_independent_reference(theta_s: float, theta_b: float) -> None:
    assert _Cs_w_Vstretching_2(_NZ, theta_s, theta_b) == pytest.approx(_ref_Vstretching_2(_NZ, theta_s, theta_b))


@pytest.mark.parametrize("theta_s", [0.65, 1.0])
@pytest.mark.parametrize("theta_b", [0.58, 1.0, 3.0])
def test_Vstretching_3_matches_independent_reference(theta_s: float, theta_b: float) -> None:
    assert _Cs_w_Vstretching_3(_NZ, theta_s, theta_b) == pytest.approx(_ref_Vstretching_3(_NZ, theta_s, theta_b))


@pytest.mark.parametrize("theta_s", [0.5, 2.0, 9.0])
@pytest.mark.parametrize("theta_b", [0.0, 1.0, 2.5])
def test_Vstretching_4_matches_independent_reference(theta_s: float, theta_b: float) -> None:
    expected = _ref_double_stretching(_ref_sigma_w_linear(_NZ), theta_s, theta_b)
    assert _Cs_w_Vstretching_4(_NZ, theta_s, theta_b) == pytest.approx(expected)


@pytest.mark.parametrize("theta_s", [0.5, 2.0, 9.0])
@pytest.mark.parametrize("theta_b", [0.0, 1.0, 2.5])
def test_Vstretching_5_matches_independent_reference(theta_s: float, theta_b: float) -> None:
    expected = _ref_double_stretching(_ref_sigma_w_quadratic(_NZ), theta_s, theta_b)
    assert _Cs_w_Vstretching_5(_NZ, theta_s, theta_b) == pytest.approx(expected)


# --- anti-drift: Vstretching=4/5 share _C_double_stretching, fed linear/quadratic sigma_w ---


def test_Vstretching_4_uses_linear_sigma_w() -> None:
    theta_s, theta_b = 3.0, 1.2
    expected = _C_double_stretching(_sigma_w_array(_NZ, "linear"), theta_s, theta_b)
    assert _Cs_w_Vstretching_4(_NZ, theta_s, theta_b) == pytest.approx(expected)


def test_Vstretching_5_uses_quadratic_sigma_w() -> None:
    theta_s, theta_b = 3.0, 1.2
    expected = _C_double_stretching(_sigma_w_array(_NZ, "quadratic"), theta_s, theta_b)
    assert _Cs_w_Vstretching_5(_NZ, theta_s, theta_b) == pytest.approx(expected)


def test_Vstretching_5_differs_from_Vstretching_4_for_the_same_parameters() -> None:
    theta_s, theta_b = 3.0, 1.2
    Cs_w_4 = _Cs_w_Vstretching_4(_NZ, theta_s, theta_b)
    Cs_w_5 = _Cs_w_Vstretching_5(_NZ, theta_s, theta_b)
    # equal only at the shared boundary values; everywhere else quadratic sigma diverges from linear
    assert not np.allclose(Cs_w_4[1:-1], Cs_w_5[1:-1])


# --- from_Vstretching_1..5 classmethod wiring ---


@pytest.mark.parametrize(
    ("name", "expected_sigma_method"),
    list(zip(_FROM_VSTRETCHING_NAMES, ["linear", "linear", "linear", "linear", "quadratic"], strict=True)),
)
def test_from_Vstretching_builds_with_the_correct_sigma_method(name: str, expected_sigma_method: str) -> None:
    constructor = getattr(LinearVerticalTransform2, name)
    transform = constructor(NZ=_NZ, hc=5.0, theta_s=3.0, theta_b=1.0)

    assert isinstance(transform, LinearVerticalTransform2)
    assert transform.sigma_method == expected_sigma_method
    assert transform.NZ == _NZ
    assert transform.hc == 5.0


@pytest.mark.parametrize("name", _FROM_VSTRETCHING_NAMES)
def test_from_Vstretching_forwards_kwargs_to_the_concrete_constructor(name: str) -> None:
    constructor = getattr(LinearVerticalTransform2, name)
    transform = constructor(NZ=_NZ, hc=5.0, theta_s=3.0, theta_b=1.0, horizontal_stencil_half_width=2)

    assert transform.horizontal_stencil_half_width == 2


@pytest.mark.parametrize("name", _FROM_VSTRETCHING_NAMES)
def test_from_Vstretching_is_defined_on_the_abstract_base_class(name: str) -> None:
    assert hasattr(VerticalTransform, name)
