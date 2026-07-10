"""Tests for the inverse ROMS vertical-coordinate transform (z -> zidx).

Black-box tests against the reference oracle, mirroring `test_compute_z_kernel.py`, plus tests
targeting the binary-search inverse specifically: the documented boundary-clamp contract, a
minimum-size stretching array, and a degenerate/flat stretching-function segment.
"""

import copy

import numpy as np
import pytest

from offline_particles.fields import FieldData, StaticField
from offline_particles.kernels.roms import construct_compute_zidx_kernel
from offline_particles.kernels.roms.vertical_coordinate import compute_zidx_kernel_function
from offline_particles.kernels.status import INACTIVE_FLAG

from . import _reference as ref


def _run_and_get_reference_zidx(particle_properties, scalars, field_data) -> tuple[np.ndarray, np.ndarray]:
    # run the compiled kernel and the reference kernel on independent copies, return both zidx arrays
    actual_properties = copy.deepcopy(particle_properties)
    expected_properties = copy.deepcopy(particle_properties)

    compute_zidx_kernel_function(actual_properties, scalars, field_data)
    ref.reference_compute_zidx_kernel_function(expected_properties, scalars, field_data)

    return actual_properties["zidx"], expected_properties["zidx"]


def _scalars(hc: float, NZ: int) -> dict[str, np.generic]:
    return {"hc": np.float64(hc), "NZ": np.int32(NZ)}


# z values corresponding to zidx in [-0.5, 0.0, 1.5, 3.0, 3.5] with hc=5.0, h=50.5, zeta=0.5,
# and C == sigma (from the linear_C_field_data fixture) -- computed via the reference oracle so
# they exercise the same interior/extrapolation regimes as the forward-kernel tests.
@pytest.fixture
def z_values_for_linear_C(hc_nz) -> list[float]:
    hc, NZ = hc_nz
    h, zeta = 50.0, 0.5
    # vectorized form of ref.sigma_coordinate, sampled at each integer zidx in [0, NZ)
    C = (np.arange(NZ, dtype=np.float64) + 0.5) / NZ - 1.0
    return [
        float(ref.compute_z(zidx, NZ, hc, h, ref.linear_interpolation(C, zidx), zeta))
        for zidx in (-0.5, 0.0, 1.5, 3.0, 3.5)
    ]


@pytest.mark.parametrize("z_index", range(5))
def test_compute_zidx_matches_reference_with_uniform_fields_and_linear_C(
    z_index: int,
    hc_nz: tuple[float, int],
    z_values_for_linear_C: list[float],
    uniform_h_zeta_field_data,
    linear_C_field_data,
    make_particle_properties,
) -> None:
    hc, NZ = hc_nz
    z = z_values_for_linear_C[z_index]
    particle_properties = make_particle_properties(zidx=[np.nan], yidx=[1.5], xidx=[1.5], z=[z])
    field_data = {**uniform_h_zeta_field_data, **linear_C_field_data}

    actual_zidx, expected_zidx = _run_and_get_reference_zidx(particle_properties, _scalars(hc, NZ), field_data)

    assert actual_zidx[0] == pytest.approx(expected_zidx[0])


@pytest.mark.parametrize("z", [-1000.0, -22.0, -10.0, 5.0, 200.0])
def test_compute_zidx_matches_reference_with_uniform_fields_and_nonlinear_C(
    z: float,
    hc_nz: tuple[float, int],
    uniform_h_zeta_field_data,
    nonlinear_C_field_data,
    make_particle_properties,
) -> None:
    hc, NZ = hc_nz
    particle_properties = make_particle_properties(zidx=[np.nan], yidx=[1.5], xidx=[1.5], z=[z])
    field_data = {**uniform_h_zeta_field_data, **nonlinear_C_field_data}

    actual_zidx, expected_zidx = _run_and_get_reference_zidx(particle_properties, _scalars(hc, NZ), field_data)

    assert actual_zidx[0] == pytest.approx(expected_zidx[0])


def test_compute_zidx_matches_reference_with_varying_fields_and_nonlinear_C(
    hc_nz: tuple[float, int],
    varying_h_zeta_field_data,
    nonlinear_C_field_data,
    make_particle_properties,
) -> None:
    hc, NZ = hc_nz
    particle_properties = make_particle_properties(
        zidx=[np.nan, np.nan, np.nan],
        yidx=[0.5, 1.25, 2.0],
        xidx=[2.0, 1.25, 0.5],
        z=[-40.0, -20.0, 10.0],
    )
    field_data = {**varying_h_zeta_field_data, **nonlinear_C_field_data}

    actual_zidx, expected_zidx = _run_and_get_reference_zidx(particle_properties, _scalars(hc, NZ), field_data)

    assert actual_zidx == pytest.approx(expected_zidx)


@pytest.mark.parametrize("z", [-1e6, -500.0])
def test_compute_zidx_clamps_below_the_seafloor_consistently_with_reference(
    z: float,
    hc_nz: tuple[float, int],
    uniform_h_zeta_field_data,
    nonlinear_C_field_data,
    make_particle_properties,
) -> None:
    """Documents the low-side clamp contract from `_compute_Cidx_from_S`'s docstring."""
    hc, NZ = hc_nz
    particle_properties = make_particle_properties(zidx=[np.nan], yidx=[1.5], xidx=[1.5], z=[z])
    field_data = {**uniform_h_zeta_field_data, **nonlinear_C_field_data}

    actual_zidx, expected_zidx = _run_and_get_reference_zidx(particle_properties, _scalars(hc, NZ), field_data)

    assert actual_zidx[0] == pytest.approx(expected_zidx[0])
    # both should extrapolate using the *first* stretching-function cell, i.e. well below zidx=0
    assert actual_zidx[0] < 0.0


@pytest.mark.parametrize("z", [1e6, 500.0])
def test_compute_zidx_clamps_above_the_surface_consistently_with_reference(
    z: float,
    hc_nz: tuple[float, int],
    uniform_h_zeta_field_data,
    nonlinear_C_field_data,
    make_particle_properties,
) -> None:
    """Documents the high-side clamp contract from `_compute_Cidx_from_S`'s docstring."""
    hc, NZ = hc_nz
    particle_properties = make_particle_properties(zidx=[np.nan], yidx=[1.5], xidx=[1.5], z=[z])
    field_data = {**uniform_h_zeta_field_data, **nonlinear_C_field_data}

    actual_zidx, expected_zidx = _run_and_get_reference_zidx(particle_properties, _scalars(hc, NZ), field_data)

    assert actual_zidx[0] == pytest.approx(expected_zidx[0])
    # both should extrapolate using the *last* stretching-function cell, i.e. well above zidx=NZ-1
    assert actual_zidx[0] > NZ - 1.0


def test_compute_zidx_handles_minimum_size_stretching_array(hc_nz: tuple[float, int]) -> None:
    """A C array of length 2 is the minimum for which the binary search's C_size - 2 >= 0."""
    hc, _ = hc_nz
    NZ = 2
    C = np.array([-0.75, -0.25])
    h, zeta = 40.0, 0.5

    for z in (-30.0, -20.0, -15.0, 0.0, 10.0):
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([np.nan]),
            "yidx": np.array([1.5]),
            "xidx": np.array([1.5]),
            "z": np.array([z]),
        }
        scalars = _scalars(hc, NZ)
        field_data = {
            "h": FieldData(np.full((4, 4), h), (0.0, 0.0)),
            "zeta": FieldData(np.full((4, 4), zeta), (0.0, 0.0)),
            "C": FieldData(C, (0.0,)),
        }

        actual_zidx, expected_zidx = _run_and_get_reference_zidx(particle_properties, scalars, field_data)
        assert actual_zidx[0] == pytest.approx(expected_zidx[0])


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_compute_zidx_swallows_zero_division_on_degenerate_C_segment(capfd) -> None:
    """Characterization test for a genuine sharp edge, found while writing this suite.

    If two consecutive C values coincide (a degenerate/flat stretching-function segment) and hc
    is small enough that S depends effectively only on C, `S_high == S_low` inside `_zidx_from_S`,
    which triggers a `ZeroDivisionError` in the underlying (noexcept nogil) Cython function.
    Because the function is declared `noexcept`, Cython cannot propagate that exception to the
    caller: it prints an "exception ignored" message to stderr (via `PyErr_WriteUnraisable`) and
    silently leaves the output at whatever (unspecified) value was already in the register --
    *not* NaN or inf as IEEE754 float division would give. The caller has no way to detect this
    via a Python exception or a NaN check. This should NOT be replicated in a numba rewrite --
    numba float division does not perform Python-style zero-division checks, so the same
    degenerate input would propagate `inf`/`nan` naturally instead of silently returning a bogus
    finite value.
    """
    C = np.array([0.0, 0.3, 0.6, 0.9, 0.9])  # the last two stretching-function levels coincide
    NZ = 5
    hc = 0.0  # makes S == C exactly, so the coincident C values give S_high == S_low exactly
    h, zeta = 45.0, 1.0
    S = 0.9  # exactly the degenerate plateau value
    z = zeta + (zeta + h) * S

    particle_properties = {
        "status": np.zeros(1, dtype=np.uint8),
        "zidx": np.array([np.nan]),
        "yidx": np.array([1.5]),
        "xidx": np.array([1.5]),
        "z": np.array([z]),
    }
    scalars = _scalars(hc, NZ)
    field_data = {
        "h": FieldData(np.full((4, 4), h), (0.0, 0.0)),
        "zeta": FieldData(np.full((4, 4), zeta), (0.0, 0.0)),
        "C": FieldData(C, (0.0,)),
    }

    compute_zidx_kernel_function(particle_properties, scalars, field_data)
    captured = capfd.readouterr()

    assert "ZeroDivisionError" in captured.err
    # the caller gets no exception and no NaN -- just a silently wrong finite value
    assert not np.isnan(particle_properties["zidx"][0])
    assert particle_properties["zidx"][0] != pytest.approx(3.0)  # the mathematically sane answer


def test_compute_zidx_skips_inactive_particles(
    hc_nz: tuple[float, int],
    uniform_h_zeta_field_data,
    nonlinear_C_field_data,
    make_particle_properties,
) -> None:
    hc, NZ = hc_nz
    particle_properties = make_particle_properties(
        zidx=[np.nan, -999.0],
        yidx=[1.5, 1.5],
        xidx=[1.5, 1.5],
        z=[-20.0, -20.0],
        status=[0, INACTIVE_FLAG],
    )
    field_data = {**uniform_h_zeta_field_data, **nonlinear_C_field_data}

    compute_zidx_kernel_function(particle_properties, _scalars(hc, NZ), field_data)

    assert np.isfinite(particle_properties["zidx"][0])
    assert particle_properties["zidx"][1] == pytest.approx(-999.0)


def test_construct_compute_zidx_kernel_honours_custom_bindings(
    hc_nz: tuple[float, int],
    uniform_h_zeta_field_data,
    linear_C_field_data,
    make_particle_properties,
    run_bound_kernel,
) -> None:
    hc, NZ = hc_nz
    bound_kernel = construct_compute_zidx_kernel(z="my_z", hc="my_hc", NZ="my_NZ", h="my_h", zeta="my_zeta", C="my_C")

    base_particle_properties = make_particle_properties(zidx=[np.nan], yidx=[1.5], xidx=[1.5], z=[-20.0])
    particle_properties = {
        "status": base_particle_properties["status"],
        "zidx": base_particle_properties["zidx"],
        "yidx": base_particle_properties["yidx"],
        "xidx": base_particle_properties["xidx"],
        "my_z": base_particle_properties["z"],
    }
    scalars = {"my_hc": np.float64(hc), "my_NZ": np.int32(NZ)}
    field_data = {
        "my_h": uniform_h_zeta_field_data["h"],
        "my_zeta": uniform_h_zeta_field_data["zeta"],
        "my_C": linear_C_field_data["C"],
    }

    _, expected_zidx = _run_and_get_reference_zidx(
        base_particle_properties, _scalars(hc, NZ), {**uniform_h_zeta_field_data, **linear_C_field_data}
    )

    run_bound_kernel(bound_kernel, particle_properties, scalars, field_data)

    assert particle_properties["zidx"][0] == pytest.approx(expected_zidx[0])


def test_construct_compute_zidx_kernel_rejects_zeta_field_with_wrong_axis_ordering() -> None:
    bound_kernel = construct_compute_zidx_kernel()
    bad_zeta_field = StaticField.from_arraylike(np.zeros((3, 3)), axes=("X", "Y"), staggers=("center", "center"))

    with pytest.raises(ValueError):
        bound_kernel.kernel.field_data["zeta"].validate_field(bad_zeta_field)


def test_construct_compute_zidx_kernel_rejects_C_field_with_wrong_axis_ordering() -> None:
    bound_kernel = construct_compute_zidx_kernel()
    bad_C_field = StaticField.from_arraylike(np.zeros((3, 3)), axes=("Z", "Y"), staggers=("center", "center"))

    with pytest.raises(ValueError):
        bound_kernel.kernel.field_data["C"].validate_field(bad_C_field)
