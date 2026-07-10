"""Tests for the forward ROMS vertical-coordinate transform (zidx -> z).

These are black-box tests: they call the compiled `compute_z_kernel_function` (and the public
`construct_compute_z_kernel` factory) and check the results against the pure-Python reference
oracle in `_reference.py`, rather than reaching into the private Cython helpers.
"""

import copy

import numpy as np
import pytest

from offline_particles.fields import StaticField
from offline_particles.kernels.roms import construct_compute_z_kernel
from offline_particles.kernels.roms.vertical_coordinate import compute_z_kernel_function
from offline_particles.kernels.status import INACTIVE_FLAG

from . import _reference as ref


def _run_and_get_reference_z(particle_properties, scalars, field_data) -> tuple[np.ndarray, np.ndarray]:
    # run the compiled kernel and the reference kernel on independent copies, return both z arrays
    actual_properties = copy.deepcopy(particle_properties)
    expected_properties = copy.deepcopy(particle_properties)

    compute_z_kernel_function(actual_properties, scalars, field_data)
    ref.reference_compute_z_kernel_function(expected_properties, scalars, field_data)

    return actual_properties["z"], expected_properties["z"]


def _scalars(hc: float, NZ: int) -> dict[str, np.generic]:
    return {"hc": np.float64(hc), "NZ": np.int32(NZ)}


@pytest.mark.parametrize("zidx", [-0.5, 0.0, 1.5, 3.0, 3.5])
def test_compute_z_matches_reference_with_uniform_fields_and_linear_C(
    zidx: float,
    hc_nz: tuple[float, int],
    uniform_h_zeta_field_data,
    linear_C_field_data,
    make_particle_properties,
) -> None:
    hc, NZ = hc_nz
    particle_properties = make_particle_properties(zidx=[zidx], yidx=[1.5], xidx=[1.5], z=[np.nan])
    field_data = {**uniform_h_zeta_field_data, **linear_C_field_data}

    actual_z, expected_z = _run_and_get_reference_z(particle_properties, _scalars(hc, NZ), field_data)

    assert actual_z[0] == pytest.approx(expected_z[0])


@pytest.mark.parametrize("zidx", [-0.5, 0.0, 1.5, 3.0, 3.5])
def test_compute_z_matches_reference_with_uniform_fields_and_nonlinear_C(
    zidx: float,
    hc_nz: tuple[float, int],
    uniform_h_zeta_field_data,
    nonlinear_C_field_data,
    make_particle_properties,
) -> None:
    hc, NZ = hc_nz
    particle_properties = make_particle_properties(zidx=[zidx], yidx=[1.5], xidx=[1.5], z=[np.nan])
    field_data = {**uniform_h_zeta_field_data, **nonlinear_C_field_data}

    actual_z, expected_z = _run_and_get_reference_z(particle_properties, _scalars(hc, NZ), field_data)

    assert actual_z[0] == pytest.approx(expected_z[0])


def test_compute_z_matches_reference_with_varying_fields_and_nonlinear_C(
    hc_nz: tuple[float, int],
    varying_h_zeta_field_data,
    nonlinear_C_field_data,
    make_particle_properties,
) -> None:
    hc, NZ = hc_nz
    # several particles at different (y, x, zidx), exercising bilinear h/zeta interpolation and
    # linear C interpolation together
    particle_properties = make_particle_properties(
        zidx=[0.25, 1.5, 2.75],
        yidx=[0.5, 1.25, 2.0],
        xidx=[2.0, 1.25, 0.5],
        z=[np.nan, np.nan, np.nan],
    )
    field_data = {**varying_h_zeta_field_data, **nonlinear_C_field_data}

    actual_z, expected_z = _run_and_get_reference_z(particle_properties, _scalars(hc, NZ), field_data)

    assert actual_z == pytest.approx(expected_z)


def test_compute_z_skips_inactive_particles(
    hc_nz: tuple[float, int],
    uniform_h_zeta_field_data,
    nonlinear_C_field_data,
    make_particle_properties,
) -> None:
    hc, NZ = hc_nz
    particle_properties = make_particle_properties(
        zidx=[1.5, 1.5],
        yidx=[1.5, 1.5],
        xidx=[1.5, 1.5],
        z=[np.nan, -999.0],
        status=[0, INACTIVE_FLAG],
    )
    field_data = {**uniform_h_zeta_field_data, **nonlinear_C_field_data}

    compute_z_kernel_function(particle_properties, _scalars(hc, NZ), field_data)

    assert np.isfinite(particle_properties["z"][0])
    assert particle_properties["z"][1] == pytest.approx(-999.0)


def test_construct_compute_z_kernel_honours_custom_bindings(
    hc_nz: tuple[float, int],
    uniform_h_zeta_field_data,
    linear_C_field_data,
    make_particle_properties,
    run_bound_kernel,
) -> None:
    hc, NZ = hc_nz
    bound_kernel = construct_compute_z_kernel(z="my_z", hc="my_hc", NZ="my_NZ", h="my_h", zeta="my_zeta", C="my_C")

    base_particle_properties = make_particle_properties(zidx=[1.5], yidx=[1.5], xidx=[1.5], z=[np.nan])
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

    _, expected_z = _run_and_get_reference_z(
        base_particle_properties, _scalars(hc, NZ), {**uniform_h_zeta_field_data, **linear_C_field_data}
    )

    run_bound_kernel(bound_kernel, particle_properties, scalars, field_data)

    assert particle_properties["my_z"][0] == pytest.approx(expected_z[0])


def test_construct_compute_z_kernel_rejects_h_field_with_wrong_axis_ordering() -> None:
    bound_kernel = construct_compute_z_kernel()
    bad_h_field = StaticField.from_arraylike(np.zeros((3, 3)), axes=("X", "Y"), staggers=("center", "center"))

    with pytest.raises(ValueError):
        bound_kernel.kernel.field_data["h"].validate_field(bad_h_field)


def test_construct_compute_z_kernel_rejects_C_field_with_wrong_axis_ordering() -> None:
    bound_kernel = construct_compute_z_kernel()
    bad_C_field = StaticField.from_arraylike(np.zeros((3,)), axes=("Y",), staggers=("center",))

    with pytest.raises(ValueError):
        bound_kernel.kernel.field_data["C"].validate_field(bad_C_field)
