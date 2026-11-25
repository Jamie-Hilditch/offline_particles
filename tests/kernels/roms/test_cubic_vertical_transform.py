"""Tests for `CubicVerticalTransform` (monotone cubic Hermite interpolation).

`sigma` and `Cs_w` both get their node derivatives from the same construction
(`_node_derivative_array`: an unfiltered Bessel slope estimate, then a Hyman (1983) monotonicity
filter), which guarantees `d(sigma)/d(zidx) >= 0` and `d(Cs_w)/d(zidx) >= 0` everywhere - and
therefore `S` itself is monotonic everywhere, unconditionally (see the class docstring). `sigma` is
reproduced exactly off-node wherever the filter is a no-op (checked against the real closed-form
formula); `Cs_w` is only ever guaranteed monotone, not exact (checked against the independent
`_reference.py` oracle across all 5 real Vstretching formulas). The inverse has no closed form on
a cubic cell and converges via a fixed, small number of Halley iterations, so round-trip and
boundary checks use a tight but non-bit-exact tolerance except where the forward direction itself
performs zero arithmetic (the `t=0` case of `_hermite_interpolation_particle`).
"""

import warnings
from typing import Literal

import numpy as np
import pytest

from offline_particles.fields import FieldData
from offline_particles.kernels.roms._vertical_coordinate import (
    CubicVerticalTransform,
    LinearVerticalTransform,
    VerticalTransform,
    _Cs_w_Vstretching_1,
    _Cs_w_Vstretching_2,
    _Cs_w_Vstretching_3,
    _Cs_w_Vstretching_4,
    _Cs_w_Vstretching_5,
    _node_derivative_array,
    _sigma_coordinate_linear,
    _sigma_coordinate_quadratic,
    _tau_Vtransform_2,
    compute_z,
)

from . import _reference as ref


# Cs_w == sigma_w exactly (linear-in-sigma): used for construction/equality/caching/ABC tests
# that don't care about the specific numeric values
def _linear_Cs_w(NZ: int) -> np.ndarray:
    return np.arange(NZ + 1, dtype=np.float64) / NZ - 1.0


# strictly increasing, nonlinear-in-sigma, with exact boundary values Cs_w[0]=-1, Cs_w[-1]=0
def _nonlinear_Cs_w(NZ: int) -> np.ndarray:
    sigma_w = _linear_Cs_w(NZ)
    return sigma_w + 0.3 * sigma_w * (sigma_w + 1.0)


def _particle_properties(zidx: float = np.nan, z: float = np.nan) -> dict[str, np.ndarray]:
    return {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([zidx]),
        "yidx": np.array([1.5]),
        "xidx": np.array([1.5]),
        "z": np.array([z]),
    }


def _uniform_field_data(h: float, zeta: float) -> dict[str, FieldData]:
    # 4x4, not 2x2: large enough for CubicVerticalTransform's default horizontal_stencil_half_width
    # of 2 (bicubic, needs 2*N=4 points per dimension); the field is uniform, so the interpolated
    # value is the same regardless of field size or stencil order.
    return {
        "h": FieldData(np.full((4, 4), h), (0.0, 0.0)),
        "zeta": FieldData(np.full((4, 4), zeta), (0.0, 0.0)),
    }


# the hand-worked example from the design plan: NZ=4, linear sigma, hc=5.0, h=45.0, zeta=1.0,
# Vtransform=2 (tau=0.1), a Cs_w whose node derivatives exercise both the interior Bessel-mean
# branch and the boundary Hyman clamp
_HAND_NZ = 4
_HAND_CS_W = np.array([-1.0, -0.9, -0.5, -0.1, 0.0])
_HAND_HC, _HAND_H, _HAND_ZETA = 5.0, 45.0, 1.0
_HAND_ZIDX, _HAND_Z = 1.0, -31.63125


# --- construction validation (thin: the full NZ/hc/Cs_w/sigma_method/Vtransform matrix is ---
# --- already covered via the shared ABC code path in test_linear_vertical_transform.py) ---


def test_horizontal_stencil_half_width_defaults_to_2() -> None:
    transform = CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))
    assert transform.horizontal_stencil_half_width == 2


@pytest.mark.parametrize("horizontal_stencil_half_width", [0, -1])
def test_rejects_non_positive_horizontal_stencil_half_width(horizontal_stencil_half_width: int) -> None:
    with pytest.raises(ValueError, match="horizontal_stencil_half_width"):
        CubicVerticalTransform(
            NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3), horizontal_stencil_half_width=horizontal_stencil_half_width
        )


def test_halley_iterations_defaults_to_3() -> None:
    transform = CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))
    assert transform.halley_iterations == 3


@pytest.mark.parametrize("halley_iterations", [0, -1])
def test_rejects_non_positive_halley_iterations(halley_iterations: int) -> None:
    with pytest.raises(ValueError, match="halley_iterations"):
        CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3), halley_iterations=halley_iterations)


def test_rejects_Cs_w_that_is_not_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=np.array([-1.0, -0.6, -0.7, 0.0]))


def test_ABC_validation_still_runs_when_subclass_post_init_forgets_nothing() -> None:
    """Regression guard: CubicVerticalTransform.__post_init__ must call super().__post_init__().

    If it didn't, none of VerticalTransform's NZ/hc/Cs_w/sigma_method/Vtransform validation would
    run at all.
    """
    with pytest.raises(ValueError, match=r"Cs_w\[0\]"):
        bad_Cs_w = _linear_Cs_w(3)
        bad_Cs_w[0] = -0.9
        CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=np.sort(bad_Cs_w))


# --- construction never warns, even for quadratic sigma at small NZ - locks in the fix: ---
# --- _node_derivative_array's Hyman filter guarantees d(sigma)/d(zidx)>=0 and d(Cs_w)/d(zidx)>=0 ---
# --- everywhere, so S's monotonicity no longer depends on NZ at all (see the class docstring) ---


@pytest.mark.parametrize("NZ", [2, 3, 6, 9, 10, 11, 50])
def test_construction_never_warns_for_quadratic_sigma_method_regardless_of_NZ(NZ: int) -> None:
    sigma_fn = _sigma_coordinate_quadratic
    Cs_w = np.array([sigma_fn(k - 0.5, NZ) for k in range(NZ + 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CubicVerticalTransform(NZ=NZ, hc=5.0, Cs_w=Cs_w, sigma_method="quadratic")


@pytest.mark.parametrize("NZ", [2, 3, 10, 50])
def test_construction_never_warns_for_linear_sigma_method(NZ: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CubicVerticalTransform(NZ=NZ, hc=5.0, Cs_w=_linear_Cs_w(NZ), sigma_method="linear")


# --- equality is value-based (every dataclass field); instances are therefore unhashable ---


def test_equal_valued_instances_are_equal() -> None:
    a = CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))
    b = CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))

    assert a == b
    assert a is not b


@pytest.mark.parametrize(
    "other_kwargs",
    [
        pytest.param({"NZ": 4, "hc": 5.0, "Cs_w": _linear_Cs_w(4)}, id="different_NZ"),
        pytest.param({"NZ": 3, "hc": 6.0, "Cs_w": _linear_Cs_w(3)}, id="different_hc"),
        pytest.param(
            {"NZ": 3, "hc": 5.0, "Cs_w": _linear_Cs_w(3), "horizontal_stencil_half_width": 1},
            id="different_horizontal_stencil_half_width",
        ),
        pytest.param({"NZ": 3, "hc": 5.0, "Cs_w": _nonlinear_Cs_w(3)}, id="different_Cs_w"),
        pytest.param(
            {"NZ": 3, "hc": 5.0, "Cs_w": _linear_Cs_w(3), "sigma_method": "quadratic"}, id="different_sigma_method"
        ),
        pytest.param({"NZ": 3, "hc": 5.0, "Cs_w": _linear_Cs_w(3), "Vtransform": 1}, id="different_Vtransform"),
        pytest.param(
            {"NZ": 3, "hc": 5.0, "Cs_w": _linear_Cs_w(3), "halley_iterations": 6}, id="different_halley_iterations"
        ),
    ],
)
def test_instances_differing_in_any_field_are_not_equal(other_kwargs: dict) -> None:
    a = CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))
    b = CubicVerticalTransform(**other_kwargs)

    assert a != b


def test_not_equal_to_a_linear_vertical_transform_with_the_same_field_values() -> None:
    a = CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))
    b = LinearVerticalTransform(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))

    assert a != b


def test_instances_are_not_hashable() -> None:
    """Intentional per the class docstring: a content-based __hash__ was deliberately not added."""
    transform = CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))

    with pytest.raises(TypeError, match="unhashable"):
        hash(transform)


# --- ABC behavior ---


def test_cubic_vertical_transform_is_a_vertical_transform() -> None:
    transform = CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))

    assert isinstance(transform, VerticalTransform)


# --- caching is per-instance only, not cross-instance ---


@pytest.mark.parametrize("method_name", ["compute_z_kernel", "compute_zidx_kernel"])
def test_same_instance_same_only_initialising_returns_the_same_object(method_name: str) -> None:
    transform = CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))
    method = getattr(transform, method_name)

    assert method(only_initialising=False) is method(only_initialising=False)
    assert method(only_initialising=True) is method(only_initialising=True)


@pytest.mark.parametrize("method_name", ["compute_z_kernel", "compute_zidx_kernel"])
def test_same_instance_different_only_initialising_returns_different_objects(method_name: str) -> None:
    transform = CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))
    method = getattr(transform, method_name)

    assert method(only_initialising=False) is not method(only_initialising=True)


@pytest.mark.parametrize("method_name", ["compute_z_kernel", "compute_zidx_kernel"])
def test_different_instances_with_equal_values_do_not_share_a_cached_kernel(method_name: str) -> None:
    Cs_w = _linear_Cs_w(3)
    a = CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=Cs_w)
    b = CubicVerticalTransform(NZ=3, hc=5.0, Cs_w=Cs_w)

    assert getattr(a, method_name)() is not getattr(b, method_name)()


# --- sigma is reproduced exactly off-node wherever the Hyman filter is a no-op: the Bessel ---
# --- slope estimate is exact for sigma (a degree <= 2 polynomial), at every node ---


@pytest.mark.parametrize("sigma_method", ["linear", "quadratic"])
def test_z_reproduces_the_exact_closed_form_sigma_off_node(sigma_method: Literal["linear", "quadratic"]) -> None:
    # Cs_w is tabulated from the *linear* sigma formula regardless of sigma_method: its node
    # spacing is then exactly uniform, so the raw Bessel estimate exactly recovers the true
    # (constant) slope and the Hyman filter never clamps it, so C's cubic degenerates exactly back
    # to that same linear formula - isolating this test to whether sigma's own node-derivative path
    # reproduces whichever sigma_method formula, independent of C's contribution. NZ=12 is large
    # enough that sigma_method="quadratic"'s continuous formula is itself monotonic between its
    # tabulated points, so the Hyman filter is a no-op there too (below that NZ, the filter clamps
    # it and this exactness claim would not hold - see test_monotonicity_is_preserved_between_nodes
    # for the small-NZ case instead).
    NZ, hc, h, zeta = 12, 5.0, 45.0, 1.0
    sigma_fn = _sigma_coordinate_quadratic if sigma_method == "quadratic" else _sigma_coordinate_linear
    Cs_w = _linear_Cs_w(NZ)
    transform = CubicVerticalTransform(NZ=NZ, hc=hc, Cs_w=Cs_w, sigma_method=sigma_method)
    kernel = transform.compute_z_kernel()

    zidx = 2.37  # deliberately off a node
    particle_properties = _particle_properties(zidx=zidx)
    field_data = _uniform_field_data(h, zeta)
    kernel(particle_properties, {}, field_data)

    expected_sigma = sigma_fn(zidx, NZ)
    expected_C = _sigma_coordinate_linear(zidx, NZ)
    tau = _tau_Vtransform_2(hc, h)
    expected_z = compute_z(expected_sigma, tau, h, expected_C, zeta)
    assert particle_properties["z"][0] == pytest.approx(expected_z, abs=1e-10)


# --- node derivatives: hand derivation exercising both the interior Bessel-mean branch and the ---
# --- boundary Hyman clamp firing at both ends ---


def test_node_derivative_array_matches_hand_derivation_for_cs_w() -> None:
    assert _node_derivative_array(_HAND_CS_W) == pytest.approx(np.array([0.0, 0.25, 0.4, 0.25, 0.0]))


# --- hand-derived forward/inverse example (see the design plan for the full worked derivation) ---


def test_compute_z_kernel_matches_hand_derivation() -> None:
    transform = CubicVerticalTransform(NZ=_HAND_NZ, hc=_HAND_HC, Cs_w=_HAND_CS_W)
    kernel = transform.compute_z_kernel()

    particle_properties = _particle_properties(zidx=_HAND_ZIDX)
    field_data = _uniform_field_data(_HAND_H, _HAND_ZETA)
    kernel(particle_properties, {}, field_data)

    assert particle_properties["z"][0] == pytest.approx(_HAND_Z)


def test_compute_zidx_kernel_matches_hand_derivation() -> None:
    # the hand-traced Halley iteration converges from an initial ~4.4% error (seed t0=0.4562, true
    # t=0.5) to a residual around 3e-5 after 1 iteration, ~1e-14 after 2, and bit noise from
    # iteration 3 onward - the fixed 6-iteration loop converges far below the abs=1e-9 tolerance
    # used here
    transform = CubicVerticalTransform(NZ=_HAND_NZ, hc=_HAND_HC, Cs_w=_HAND_CS_W)
    kernel = transform.compute_zidx_kernel()

    particle_properties = _particle_properties(z=_HAND_Z)
    field_data = _uniform_field_data(_HAND_H, _HAND_ZETA)
    kernel(particle_properties, {}, field_data)

    assert particle_properties["zidx"][0] == pytest.approx(_HAND_ZIDX, abs=1e-9)


# --- monotonicity preservation: the interpolated stretching function never overshoots or dips ---
# --- between nodes, checked against the independent `_reference.py` oracle (not the real numba ---
# --- code) across all 5 real Vstretching formulas at a spread of valid theta_s/theta_b ---

_VSTRETCHING_CASES = [
    (_Cs_w_Vstretching_1, 3.0, 0.5),
    (_Cs_w_Vstretching_1, 8.0, 1.0),
    (_Cs_w_Vstretching_2, 3.0, 0.5),
    (_Cs_w_Vstretching_2, 10.0, 3.0),
    (_Cs_w_Vstretching_3, 0.65, 0.58),
    (_Cs_w_Vstretching_3, 1.0, 3.0),
    (_Cs_w_Vstretching_4, 5.0, 1.5),
    (_Cs_w_Vstretching_4, 10.0, 3.0),
    (_Cs_w_Vstretching_5, 5.0, 1.5),
    (_Cs_w_Vstretching_5, 10.0, 3.0),
]


@pytest.mark.parametrize("NZ", [4, 10, 32])
@pytest.mark.parametrize("Vstretching_fn, theta_s, theta_b", _VSTRETCHING_CASES)
def test_monotonicity_is_preserved_between_nodes(Vstretching_fn, theta_s: float, theta_b: float, NZ: int) -> None:
    Cs_w = Vstretching_fn(NZ, theta_s, theta_b)
    derivative = ref.node_derivative_array(Cs_w)
    t = np.linspace(0.0, 1.0, 101)

    for i in range(Cs_w.shape[0] - 1):
        f0, f1 = Cs_w[i], Cs_w[i + 1]
        a, b, c, d = ref.hermite_cubic_coefficients(f0, f1, derivative[i], derivative[i + 1])
        values = np.array([ref.cubic_eval(a, b, c, d, ti) for ti in t])
        assert np.all(np.diff(values) >= -1e-12)
        assert np.all(values >= min(f0, f1) - 1e-12)
        assert np.all(values <= max(f0, f1) + 1e-12)


@pytest.mark.parametrize("NZ", [2, 3, 4, 5, 6, 9, 10])
def test_quadratic_sigma_monotonicity_is_preserved_between_nodes_even_at_small_NZ(NZ: int) -> None:
    """The actual fix being locked in: `sigma_w` itself used to dip between nodes at small `NZ`.

    Checked directly against the independent `_reference.py` oracle, not the real numba code -
    same pattern as `test_monotonicity_is_preserved_between_nodes`, but for `sigma_w` rather than
    `Cs_w`, since `sigma_w` is the array that previously had the problem (see the class docstring).
    """
    sigma_w = ref.sigma_w_array(NZ, "quadratic")
    derivative = ref.node_derivative_array(sigma_w)
    t = np.linspace(0.0, 1.0, 101)

    for i in range(sigma_w.shape[0] - 1):
        f0, f1 = sigma_w[i], sigma_w[i + 1]
        a, b, c, d = ref.hermite_cubic_coefficients(f0, f1, derivative[i], derivative[i + 1])
        values = np.array([ref.cubic_eval(a, b, c, d, ti) for ti in t])
        assert np.all(np.diff(values) >= -1e-12)
        assert np.all(values >= min(f0, f1) - 1e-12)
        assert np.all(values <= max(f0, f1) + 1e-12)


# --- round-trip, both directions, across Vtransform and sigma_method - empirical (abs=1e-9), ---
# --- not provably exact: the inverse converges via a fixed, small number of Halley iterations, ---
# --- not a closed form (see the class docstring) ---


@pytest.mark.parametrize("sigma_method", ["linear", "quadratic"])
@pytest.mark.parametrize("Vtransform", [1, 2])
def test_compute_z_and_compute_zidx_kernels_round_trip(
    Vtransform: Literal[1, 2], sigma_method: Literal["linear", "quadratic"]
) -> None:
    NZ, hc, h, zeta, zidx = 12, 5.0, 45.0, 1.0, 2.5
    Cs_w = _nonlinear_Cs_w(NZ)
    transform = CubicVerticalTransform(NZ=NZ, hc=hc, Cs_w=Cs_w, sigma_method=sigma_method, Vtransform=Vtransform)

    particle_properties = _particle_properties(zidx=zidx)
    field_data = _uniform_field_data(h, zeta)
    transform.compute_z_kernel()(particle_properties, {}, field_data)
    transform.compute_zidx_kernel()(particle_properties, {}, field_data)

    assert particle_properties["zidx"][0] == pytest.approx(zidx, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("NZ", [2, 3, 6, 9])
@pytest.mark.parametrize("Vtransform", [1, 2])
def test_compute_z_and_compute_zidx_kernels_round_trip_with_quadratic_sigma_at_small_NZ(
    Vtransform: Literal[1, 2], NZ: int
) -> None:
    """Locks in the actual fix: this combination used to warn and could round-trip incorrectly."""
    hc, h, zeta = 5.0, 50.0, 1.0
    Cs_w = _nonlinear_Cs_w(NZ)
    transform = CubicVerticalTransform(NZ=NZ, hc=hc, Cs_w=Cs_w, sigma_method="quadratic", Vtransform=Vtransform)

    for zidx in np.linspace(0.0, NZ - 1.0, 13):
        particle_properties = _particle_properties(zidx=zidx)
        field_data = _uniform_field_data(h, zeta)
        transform.compute_z_kernel()(particle_properties, {}, field_data)
        transform.compute_zidx_kernel()(particle_properties, {}, field_data)
        assert particle_properties["zidx"][0] == pytest.approx(zidx, rel=1e-9, abs=1e-9)


# --- halley_iterations is a per-instance parameter, not a module constant, precisely because its ---
# --- need depends on how close the seed starts to the true root - worst case near tau's maximum ---
# --- of 1 (Vtransform=1, h close to hc), not on typical grids. The default (3) is chosen for ---
# --- typical grids, not for this edge case; users who hit it can raise halley_iterations instead ---
# --- of the library paying for it everywhere. ---


def test_halley_iterations_default_is_insufficient_right_at_tau_approx_1_but_more_fixes_it() -> None:
    """A concrete, reproducible case near tau's maximum, found by a dense (non-random) sweep.

    `h` is `hc` plus a relative `1e-9`, i.e. `tau = hc/h` is as close to 1 as a typical grid would
    ever realistically get without hitting it exactly (`hc <= h` is `LinearVerticalTransform` and
    `VerticalTransform`'s standing contract for `Vtransform=1`). This is deliberately *not* what
    sets the class default - see the class docstring.
    """
    NZ, hc = 7, 0.5
    h = hc * (1.0 + 1e-9)
    zeta = h * -0.4
    zidx = 6.0
    Cs_w = _nonlinear_Cs_w(NZ)

    default_transform = CubicVerticalTransform(NZ=NZ, hc=hc, Cs_w=Cs_w, sigma_method="quadratic", Vtransform=1)
    particle_properties = _particle_properties(zidx=zidx)
    field_data = _uniform_field_data(h, zeta)
    default_transform.compute_z_kernel()(particle_properties, {}, field_data)
    default_transform.compute_zidx_kernel()(particle_properties, {}, field_data)
    default_error = abs(particle_properties["zidx"][0] - zidx)
    assert default_error > 1e-6  # the default (3) is NOT enough precision for this edge case

    precise_transform = CubicVerticalTransform(
        NZ=NZ, hc=hc, Cs_w=Cs_w, sigma_method="quadratic", Vtransform=1, halley_iterations=6
    )
    particle_properties = _particle_properties(zidx=zidx)
    precise_transform.compute_z_kernel()(particle_properties, {}, field_data)
    precise_transform.compute_zidx_kernel()(particle_properties, {}, field_data)
    assert particle_properties["zidx"][0] == pytest.approx(zidx, abs=1e-9)  # bumping the parameter fixes it


# --- boundary exactness, asymmetry made explicit: t=0 performs zero arithmetic (d=f0), so the ---
# --- forward value is bit-exact for any Cs_w/sigma_method/tau; t=1 relies on a+b+c+d -> f1, ---
# --- exact symbolically but only to a few ULP in float64 in general - not structurally guaranteed ---


def test_z_at_the_lower_w_point_boundary_is_bit_exact() -> None:
    transform = CubicVerticalTransform(NZ=_HAND_NZ, hc=_HAND_HC, Cs_w=_HAND_CS_W)
    kernel = transform.compute_z_kernel()

    particle_properties = _particle_properties(zidx=-0.5)
    field_data = _uniform_field_data(_HAND_H, _HAND_ZETA)
    kernel(particle_properties, {}, field_data)

    assert particle_properties["z"][0] == -_HAND_H  # exact: Cs_w[0] == sigma_w[0] == -1, any tau


def test_z_at_the_upper_w_point_boundary_is_not_always_bit_exact() -> None:
    NZ, hc, h, zeta = 3, 5.0, 45.0, 1.0
    transform = CubicVerticalTransform(NZ=NZ, hc=hc, Cs_w=_nonlinear_Cs_w(NZ))
    kernel = transform.compute_z_kernel()

    particle_properties = _particle_properties(zidx=NZ - 0.5)
    field_data = _uniform_field_data(h, zeta)
    kernel(particle_properties, {}, field_data)

    assert particle_properties["z"][0] == pytest.approx(zeta)  # exact to within float64 rounding
    assert particle_properties["z"][0] != zeta  # the point being tested: not bit-exact


def test_round_trip_at_every_w_point_node_is_accurate_to_a_few_ulp() -> None:
    """A dedicated robustness check, independent of `_CUBIC_HALLEY_ITERATIONS`.

    At a node that is `t=0` of its cell, `g(0) == 0` exactly given a bit-exact `S`, so Halley's
    update leaves `t=0` unchanged going back. In practice the round trip is not quite bit-exact at
    every node (the `z`/`S` conversions around it introduce a few ULP of rounding), but it stays
    far tighter than the generic round-trip tolerance used elsewhere in this file.
    """
    transform = CubicVerticalTransform(NZ=_HAND_NZ, hc=_HAND_HC, Cs_w=_HAND_CS_W)

    for node_zidx in np.arange(_HAND_NZ + 1, dtype=np.float64) - 0.5:
        particle_properties = _particle_properties(zidx=node_zidx)
        field_data = _uniform_field_data(_HAND_H, _HAND_ZETA)
        transform.compute_z_kernel()(particle_properties, {}, field_data)
        transform.compute_zidx_kernel()(particle_properties, {}, field_data)
        assert particle_properties["zidx"][0] == pytest.approx(node_zidx, abs=1e-12)
