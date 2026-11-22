"""Tests for `LinearVerticalTransform2` (issue #147 refactor, stages 1-2).

`Cs_w` (the stretching function at w points, matching ROMS' own output variable name) is captured
directly by the dataclass and closed over by the compiled kernels, rather than fetched per-call as
windowed field data - these tests check construction validation, the deliberately-chosen
per-instance-only caching behavior, numerical correctness against the existing scalar helpers
(using the fixed `C_offset=0.5` that maps a continuous rho-point `zidx` onto the w-point array),
and that the original window-bracketing bug (issue #147) is structurally impossible here, since
there is no window at all - `Cs_w` is always the complete array.
"""

import numpy as np
import pytest

from offline_particles.fields import FieldData
from offline_particles.kernels.interpolation import bicubic_interpolation_particle
from offline_particles.kernels.roms._vertical_coordinate import LinearVerticalTransform2, compute_z, compute_zidx
from offline_particles.kernels.status import INACTIVE_FLAG, Status


# Cs_w == sigma_w exactly: degenerates the S-transform to plain sigma coordinates
def _linear_Cs_w(NZ: int) -> np.ndarray:
    return np.arange(NZ + 1, dtype=np.float64) / NZ - 1.0


# strictly increasing, nonlinear-in-sigma, with exact boundary values Cs_w[0]=-1, Cs_w[-1]=0
def _nonlinear_Cs_w(NZ: int) -> np.ndarray:
    sigma_w = _linear_Cs_w(NZ)
    return sigma_w + 0.3 * sigma_w * (sigma_w + 1.0)


# --- construction validation ---


def test_rejects_NZ_below_2() -> None:
    with pytest.raises(ValueError, match="NZ"):
        LinearVerticalTransform2(NZ=1, hc=5.0, Cs_w=_linear_Cs_w(1))


@pytest.mark.parametrize("hc", [0.0, -1.0])
def test_rejects_non_positive_hc(hc: float) -> None:
    with pytest.raises(ValueError, match="hc"):
        LinearVerticalTransform2(NZ=2, hc=hc, Cs_w=_linear_Cs_w(2))


def test_rejects_Cs_w_with_wrong_length() -> None:
    with pytest.raises(ValueError, match="Cs_w"):
        LinearVerticalTransform2(NZ=4, hc=5.0, Cs_w=_linear_Cs_w(3))  # NZ+1 != 4+1


def test_rejects_Cs_w_with_wrong_ndim() -> None:
    with pytest.raises(ValueError, match="Cs_w"):
        LinearVerticalTransform2(NZ=4, hc=5.0, Cs_w=np.zeros((2, 2)))


def test_rejects_Cs_w_that_is_not_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=np.array([-1.0, -0.6, -0.7, 0.0]))


@pytest.mark.parametrize("bad_first_value", [-0.9, -1.1])
def test_rejects_Cs_w_with_wrong_lower_boundary(bad_first_value: float) -> None:
    Cs_w = _linear_Cs_w(3)
    Cs_w[0] = bad_first_value
    with pytest.raises(ValueError, match=r"Cs_w\[0\]"):
        LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=np.sort(Cs_w))


@pytest.mark.parametrize("bad_last_value", [0.1, -0.1])
def test_rejects_Cs_w_with_wrong_upper_boundary(bad_last_value: float) -> None:
    Cs_w = _linear_Cs_w(3)
    Cs_w[-1] = bad_last_value
    with pytest.raises(ValueError, match=r"Cs_w\[-1\]"):
        LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=np.sort(Cs_w))


def test_Cs_w_is_defensively_copied_and_read_only() -> None:
    original = _linear_Cs_w(3)
    transform = LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=original)

    assert transform.Cs_w is not original
    assert not transform.Cs_w.flags.writeable
    original[1] = 999.0
    assert transform.Cs_w[1] == pytest.approx(-2.0 / 3.0)


def test_horizontal_stencil_half_width_defaults_to_1() -> None:
    transform = LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))

    assert transform.horizontal_stencil_half_width == 1


@pytest.mark.parametrize("horizontal_stencil_half_width", [0, -1])
def test_rejects_non_positive_horizontal_stencil_half_width(horizontal_stencil_half_width: int) -> None:
    with pytest.raises(ValueError, match="horizontal_stencil_half_width"):
        LinearVerticalTransform2(
            NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3), horizontal_stencil_half_width=horizontal_stencil_half_width
        )


# --- the w-point boundary values are interpolated exactly, not extrapolated ---


def test_z_at_the_exact_seafloor_and_surface_matches_minus_h_and_zeta() -> None:
    NZ, hc, h, zeta = 20, 5.0, 45.0, 1.0
    transform = LinearVerticalTransform2(NZ=NZ, hc=hc, Cs_w=_nonlinear_Cs_w(NZ))
    kernel = transform.compute_z_kernel()

    def _z_at(zidx: float) -> float:
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([zidx]),
            "yidx": np.array([1.5]),
            "xidx": np.array([1.5]),
            "z": np.array([np.nan]),
        }
        field_data = {
            "h": FieldData(np.full((2, 2), h), (0.0, 0.0)),
            "zeta": FieldData(np.full((2, 2), zeta), (0.0, 0.0)),
        }
        kernel(particle_properties, {}, field_data)
        return particle_properties["z"][0]

    assert _z_at(-0.5) == pytest.approx(-h)  # the seafloor, Cs_w[0] == -1 exactly
    assert _z_at(NZ - 0.5) == pytest.approx(zeta)  # the surface, Cs_w[-1] == 0 exactly


# --- horizontal_stencil_half_width actually changes the h/zeta interpolation used ---


def test_horizontal_stencil_half_width_2_matches_bicubic_interpolation_particle() -> None:
    NZ, hc = 3, 5.0
    Cs_w = _linear_Cs_w(NZ)
    # h/zeta with curvature, so N=1 (bilinear) and N=2 (bicubic) genuinely differ
    ii = np.arange(6, dtype=np.float64)[:, None]
    jj = np.arange(6, dtype=np.float64)[None, :]
    h_array = 40.0 + 2.0 * ii**2 + 0.5 * jj
    zeta_array = np.full((6, 6), 0.5) + 0.1 * jj**2
    yidx, xidx, zidx = np.float64(2.3), np.float64(2.7), np.float64(1.0)

    transform = LinearVerticalTransform2(NZ=NZ, hc=hc, Cs_w=Cs_w, horizontal_stencil_half_width=2)
    kernel = transform.compute_z_kernel()
    particle_properties = {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([zidx]),
        "yidx": np.array([yidx]),
        "xidx": np.array([xidx]),
        "z": np.array([np.nan]),
    }
    field_data = {"h": FieldData(h_array, (0.0, 0.0)), "zeta": FieldData(zeta_array, (0.0, 0.0))}
    kernel(particle_properties, {}, field_data)

    expected_h = bicubic_interpolation_particle(h_array, yidx, xidx, h_array.shape[0] - 4, h_array.shape[1] - 4)
    expected_zeta = bicubic_interpolation_particle(
        zeta_array, yidx, xidx, zeta_array.shape[0] - 4, zeta_array.shape[1] - 4
    )
    expected_C_value = np.interp(zidx + 0.5, np.arange(NZ + 1, dtype=np.float64), Cs_w)
    expected_z = compute_z(zidx, NZ, hc, expected_h, expected_C_value, expected_zeta)
    assert particle_properties["z"][0] == pytest.approx(expected_z)


def test_horizontal_stencil_half_width_1_and_2_give_different_results_for_a_curved_field() -> None:
    NZ, hc = 3, 5.0
    Cs_w = _linear_Cs_w(NZ)
    ii = np.arange(6, dtype=np.float64)[:, None]
    jj = np.arange(6, dtype=np.float64)[None, :]
    h_array = 40.0 + 2.0 * ii**2 + 0.5 * jj
    zeta_array = np.full((6, 6), 0.5)
    particle_properties_template = {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([1.0]),
        "yidx": np.array([2.3]),
        "xidx": np.array([2.7]),
        "z": np.array([np.nan]),
    }
    field_data = {"h": FieldData(h_array, (0.0, 0.0)), "zeta": FieldData(zeta_array, (0.0, 0.0))}

    pp1 = {k: np.array(v) for k, v in particle_properties_template.items()}
    LinearVerticalTransform2(NZ=NZ, hc=hc, Cs_w=Cs_w, horizontal_stencil_half_width=1).compute_z_kernel()(
        pp1, {}, field_data
    )
    pp2 = {k: np.array(v) for k, v in particle_properties_template.items()}
    LinearVerticalTransform2(NZ=NZ, hc=hc, Cs_w=Cs_w, horizontal_stencil_half_width=2).compute_z_kernel()(
        pp2, {}, field_data
    )

    assert pp1["z"][0] != pytest.approx(pp2["z"][0])


def test_rejects_h_field_too_small_for_the_stencil() -> None:
    NZ, hc = 3, 5.0
    Cs_w = _linear_Cs_w(NZ)
    transform = LinearVerticalTransform2(NZ=NZ, hc=hc, Cs_w=Cs_w, horizontal_stencil_half_width=2)
    kernel = transform.compute_z_kernel()

    particle_properties = {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([1.0]),
        "yidx": np.array([1.5]),
        "xidx": np.array([1.5]),
        "z": np.array([np.nan]),
    }
    # only 3x3, but horizontal_stencil_half_width=2 needs at least 4x4
    field_data = {
        "h": FieldData(np.full((3, 3), 50.0), (0.0, 0.0)),
        "zeta": FieldData(np.full((4, 4), 0.5), (0.0, 0.0)),
    }
    with pytest.raises(ValueError, match="horizontal_stencil_half_width"):
        kernel(particle_properties, {}, field_data)


# --- equality is value-based (NZ, hc, horizontal_stencil_half_width, every element of Cs_w); ---
# --- instances are therefore unhashable, since __eq__ is defined without a matching __hash__ ---


def test_equal_valued_instances_are_equal() -> None:
    a = LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))
    b = LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))

    assert a == b
    assert a is not b


@pytest.mark.parametrize(
    "other_kwargs",
    [
        {"NZ": 4, "hc": 5.0, "Cs_w": _linear_Cs_w(4)},
        {"NZ": 3, "hc": 6.0, "Cs_w": _linear_Cs_w(3)},
        {"NZ": 3, "hc": 5.0, "Cs_w": _linear_Cs_w(3), "horizontal_stencil_half_width": 2},
        {"NZ": 3, "hc": 5.0, "Cs_w": _nonlinear_Cs_w(3)},
    ],
    ids=["different_NZ", "different_hc", "different_horizontal_stencil_half_width", "different_Cs_w"],
)
def test_instances_differing_in_any_field_are_not_equal(other_kwargs: dict) -> None:
    a = LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))
    b = LinearVerticalTransform2(**other_kwargs)

    assert a != b


def test_not_equal_to_an_unrelated_object() -> None:
    a = LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))

    assert a != "not a transform"


def test_instances_are_not_hashable() -> None:
    """Intentional per the class docstring: a content-based __hash__ was deliberately not added."""
    transform = LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))

    with pytest.raises(TypeError, match="unhashable"):
        hash(transform)


# --- caching is per-instance only, not cross-instance ---


@pytest.mark.parametrize("method_name", ["compute_z_kernel", "compute_zidx_kernel"])
def test_same_instance_same_only_initialising_returns_the_same_object(method_name: str) -> None:
    transform = LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))
    method = getattr(transform, method_name)

    assert method(only_initialising=False) is method(only_initialising=False)
    assert method(only_initialising=True) is method(only_initialising=True)


@pytest.mark.parametrize("method_name", ["compute_z_kernel", "compute_zidx_kernel"])
def test_same_instance_different_only_initialising_returns_different_objects(method_name: str) -> None:
    transform = LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=_linear_Cs_w(3))
    method = getattr(transform, method_name)

    assert method(only_initialising=False) is not method(only_initialising=True)


@pytest.mark.parametrize("method_name", ["compute_z_kernel", "compute_zidx_kernel"])
def test_different_instances_with_equal_values_do_not_share_a_cached_kernel(method_name: str) -> None:
    """Documents the trade-off: no cross-instance caching, unlike the old @cache'd hc/NZ factories."""
    Cs_w = _linear_Cs_w(3)
    a = LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=Cs_w)
    b = LinearVerticalTransform2(NZ=3, hc=5.0, Cs_w=Cs_w)

    assert getattr(a, method_name)() is not getattr(b, method_name)()


# --- numerical correctness, against the existing scalar helpers ---


def test_compute_z_kernel_matches_hand_derivation() -> None:
    # NZ=4, hc=5.0, h=45.0, zeta=1.0, zidx=1.5, Cs_w == sigma_w -> local index zidx+0.5 == 2.0
    # exactly, Cs_w[2] == -0.5 -> z == -22.0 (same hand-derived case as
    # test_vertical_coordinate_helpers.py, since Cs_w==sigma_w degenerates identically to C==sigma)
    NZ, hc, h, zeta, zidx = 4, 5.0, 45.0, 1.0, 1.5
    transform = LinearVerticalTransform2(NZ=NZ, hc=hc, Cs_w=_linear_Cs_w(NZ))
    kernel = transform.compute_z_kernel()

    particle_properties = {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([zidx]),
        "yidx": np.array([1.5]),
        "xidx": np.array([1.5]),
        "z": np.array([np.nan]),
    }
    field_data = {
        "h": FieldData(np.full((2, 2), h), (0.0, 0.0)),
        "zeta": FieldData(np.full((2, 2), zeta), (0.0, 0.0)),
    }
    kernel(particle_properties, {}, field_data)

    assert particle_properties["z"][0] == pytest.approx(-22.0)


@pytest.mark.parametrize("z", [-1000.0, -22.0, -10.0, 5.0, 200.0])
def test_compute_zidx_kernel_matches_scalar_helper_with_nonlinear_Cs_w(z: float) -> None:
    NZ, hc, h, zeta = 4, 5.0, 50.0, 0.5
    Cs_w = _nonlinear_Cs_w(NZ)
    transform = LinearVerticalTransform2(NZ=NZ, hc=hc, Cs_w=Cs_w)
    kernel = transform.compute_zidx_kernel()

    particle_properties = {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([np.nan]),
        "yidx": np.array([1.5]),
        "xidx": np.array([1.5]),
        "z": np.array([z]),
    }
    field_data = {
        "h": FieldData(np.full((4, 4), h), (0.3, -0.2)),
        "zeta": FieldData(np.full((4, 4), zeta), (-0.1, 0.15)),
    }
    kernel(particle_properties, {}, field_data)

    expected_zidx = compute_zidx(z, h, zeta, hc, NZ, Cs_w, 0.5)
    assert particle_properties["zidx"][0] == pytest.approx(expected_zidx)


def test_compute_z_and_compute_zidx_kernels_round_trip() -> None:
    NZ, hc, h, zeta, zidx = 6, 5.0, 45.0, 1.0, 2.5
    Cs_w = _nonlinear_Cs_w(NZ)
    transform = LinearVerticalTransform2(NZ=NZ, hc=hc, Cs_w=Cs_w)

    particle_properties = {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([zidx]),
        "yidx": np.array([1.5]),
        "xidx": np.array([1.5]),
        "z": np.array([np.nan]),
    }
    field_data = {
        "h": FieldData(np.full((2, 2), h), (0.0, 0.0)),
        "zeta": FieldData(np.full((2, 2), zeta), (0.0, 0.0)),
    }
    transform.compute_z_kernel()(particle_properties, {}, field_data)
    transform.compute_zidx_kernel()(particle_properties, {}, field_data)

    assert particle_properties["zidx"][0] == pytest.approx(zidx)


def test_only_initialising_updates_initialising_particles_only() -> None:
    NZ, hc, h, zeta = 4, 5.0, 50.0, 0.5
    Cs_w = _nonlinear_Cs_w(NZ)
    transform = LinearVerticalTransform2(NZ=NZ, hc=hc, Cs_w=Cs_w)
    kernel = transform.compute_zidx_kernel(only_initialising=True)

    particle_properties = {
        "status": np.array(
            [np.uint8(Status.NORMAL), np.uint8(Status.INACTIVE), np.uint8(Status.INITIALISING)], dtype=np.uint8
        ),
        "zidx": np.array([-111.0, -222.0, np.nan]),
        "yidx": np.array([1.5, 1.5, 1.5]),
        "xidx": np.array([1.5, 1.5, 1.5]),
        "z": np.array([-20.0, -20.0, -20.0]),
    }
    field_data = {
        "h": FieldData(np.full((4, 4), h), (0.3, -0.2)),
        "zeta": FieldData(np.full((4, 4), zeta), (-0.1, 0.15)),
    }
    kernel(particle_properties, {}, field_data)

    assert particle_properties["zidx"][0] == pytest.approx(-111.0)
    assert particle_properties["zidx"][1] == pytest.approx(-222.0)
    assert np.isfinite(particle_properties["zidx"][2])


def test_skips_inactive_particles() -> None:
    NZ, hc, h, zeta = 4, 5.0, 50.0, 0.5
    Cs_w = _nonlinear_Cs_w(NZ)
    transform = LinearVerticalTransform2(NZ=NZ, hc=hc, Cs_w=Cs_w)
    kernel = transform.compute_zidx_kernel()

    particle_properties = {
        "status": np.array([0, INACTIVE_FLAG], dtype=np.uint8),
        "zidx": np.array([np.nan, -999.0]),
        "yidx": np.array([1.5, 1.5]),
        "xidx": np.array([1.5, 1.5]),
        "z": np.array([-20.0, -20.0]),
    }
    field_data = {
        "h": FieldData(np.full((4, 4), h), (0.3, -0.2)),
        "zeta": FieldData(np.full((4, 4), zeta), (-0.1, 0.15)),
    }
    kernel(particle_properties, {}, field_data)

    assert np.isfinite(particle_properties["zidx"][0])
    assert particle_properties["zidx"][1] == pytest.approx(-999.0)


# --- the actual fix: no window at all, so the original issue #147 bracketing bug is impossible ---


def test_computes_the_correct_zidx_far_from_index_zero_with_no_bracketing_window_at_all() -> None:
    """Issue #147: no window means no bracketing bug is possible, regardless of placeholder zidx."""
    HC, H, ZETA, NZ = 5.0, 45.0, 1.0, 20
    Cs_w = -1.0 + (np.arange(NZ + 1, dtype=np.float64) / NZ) ** 2  # exact boundaries, strictly increasing
    z = -30.0  # true zidx far from the zero-fill placeholder; old windowed approach got it wrong
    transform = LinearVerticalTransform2(NZ=NZ, hc=HC, Cs_w=Cs_w)
    kernel = transform.compute_zidx_kernel()

    particle_properties = {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([0.0]),  # the problematic placeholder value from Particles.__init__
        "yidx": np.array([1.5]),
        "xidx": np.array([1.5]),
        "z": np.array([z]),
    }
    field_data = {
        "h": FieldData(np.full((2, 2), H), (0.0, 0.0)),
        "zeta": FieldData(np.full((2, 2), ZETA), (0.0, 0.0)),
    }
    kernel(particle_properties, {}, field_data)

    expected_zidx = compute_zidx(z, H, ZETA, HC, NZ, Cs_w, 0.5)
    assert expected_zidx > 8  # sanity-check: this would NOT have been bracketed by a small window
    assert particle_properties["zidx"][0] == pytest.approx(expected_zidx)


def test_compute_z_matches_scalar_helper_used_directly() -> None:
    """Sanity-check that the kernel's per-particle math is literally `compute_z`, unchanged."""
    NZ, hc, h, zeta, zidx = 10, 7.5, 60.0, 0.8, 4.25
    Cs_w = _nonlinear_Cs_w(NZ)
    transform = LinearVerticalTransform2(NZ=NZ, hc=hc, Cs_w=Cs_w)
    kernel = transform.compute_z_kernel()

    particle_properties = {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([zidx]),
        "yidx": np.array([1.5]),
        "xidx": np.array([1.5]),
        "z": np.array([np.nan]),
    }
    field_data = {
        "h": FieldData(np.full((2, 2), h), (0.0, 0.0)),
        "zeta": FieldData(np.full((2, 2), zeta), (0.0, 0.0)),
    }
    kernel(particle_properties, {}, field_data)

    C_value = np.interp(zidx + 0.5, np.arange(NZ + 1, dtype=np.float64), Cs_w)
    expected_z = compute_z(zidx, NZ, hc, h, C_value, zeta)
    assert particle_properties["z"][0] == pytest.approx(expected_z)
