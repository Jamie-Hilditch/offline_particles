"""Tests for the `construct_compute_z_kernel`/`construct_compute_zidx_kernel` public API.

These check the *wiring* (declarations, bindings, and that custom bindings actually work when the
kernel runs) rather than the vertical-coordinate math itself, which is covered by
`test_linear_vertical_transform_2.py`.
"""

import numpy as np
import pytest

from offline_particles.fields import StaticField
from offline_particles.kernels._kernels import BoundKernel
from offline_particles.kernels.roms import (
    LinearVerticalTransform2,
    construct_compute_z_kernel,
    construct_compute_zidx_kernel,
)

_CONSTRUCTORS = (construct_compute_z_kernel, construct_compute_zidx_kernel)


@pytest.mark.parametrize("constructor", _CONSTRUCTORS)
def test_constructor_uses_declared_names_as_default_bindings(
    constructor, linear_vertical_transform: LinearVerticalTransform2
) -> None:
    bound_kernel = constructor(linear_vertical_transform)

    assert isinstance(bound_kernel, BoundKernel)
    for declared_name, bound_name in bound_kernel.particle_property_bindings.items():
        assert bound_name == declared_name
    for declared_name, bound_name in bound_kernel.scalar_bindings.items():
        assert bound_name == declared_name
    for declared_name, bound_name in bound_kernel.field_data_bindings.items():
        assert bound_name == declared_name


@pytest.mark.parametrize("constructor", _CONSTRUCTORS)
def test_constructor_honours_custom_bindings(constructor, linear_vertical_transform: LinearVerticalTransform2) -> None:
    bound_kernel = constructor(linear_vertical_transform, z="my_z", h="my_h", zeta="my_zeta")

    assert bound_kernel.particle_property_bindings["z"] == "my_z"
    assert bound_kernel.field_data_bindings["h"] == "my_h"
    assert bound_kernel.field_data_bindings["zeta"] == "my_zeta"


@pytest.mark.parametrize("constructor", _CONSTRUCTORS)
def test_constructor_declares_the_full_shared_input_set(
    constructor, linear_vertical_transform: LinearVerticalTransform2
) -> None:
    bound_kernel = constructor(linear_vertical_transform)

    assert set(bound_kernel.kernel.particle_properties) == {"status", "zidx", "yidx", "xidx", "z"}
    assert set(bound_kernel.kernel.scalars) == set()
    assert set(bound_kernel.kernel.field_data) == {"h", "zeta"}


@pytest.mark.parametrize("constructor", _CONSTRUCTORS)
def test_constructor_forwards_only_initialising(
    constructor, linear_vertical_transform: LinearVerticalTransform2
) -> None:
    default_kernel = constructor(linear_vertical_transform)
    initialising_kernel = constructor(linear_vertical_transform, only_initialising=True)

    assert default_kernel.kernel is not initialising_kernel.kernel


def test_both_constructors_declare_identical_inputs(linear_vertical_transform: LinearVerticalTransform2) -> None:
    """`roms_ab3_timestepper` shares bindings between the two kernels, relying on this implicitly."""
    z_kernel = construct_compute_z_kernel(linear_vertical_transform).kernel
    zidx_kernel = construct_compute_zidx_kernel(linear_vertical_transform).kernel

    assert set(z_kernel.particle_properties) == set(zidx_kernel.particle_properties)
    assert set(z_kernel.scalars) == set(zidx_kernel.scalars)
    assert set(z_kernel.field_data) == set(zidx_kernel.field_data)


@pytest.mark.parametrize("constructor", _CONSTRUCTORS)
def test_constructor_rejects_h_field_with_wrong_axis_ordering(
    constructor, linear_vertical_transform: LinearVerticalTransform2
) -> None:
    bound_kernel = constructor(linear_vertical_transform)
    bad_h_field = StaticField.from_arraylike(np.zeros((3, 3)), axes=("X", "Y"), staggers=("center", "center"))

    with pytest.raises(ValueError):
        bound_kernel.kernel.field_data["h"].validate_field(bad_h_field)


def test_construct_compute_z_kernel_runs_correctly_with_custom_bindings(
    linear_vertical_transform: LinearVerticalTransform2,
    uniform_h_zeta_field_data,
    make_particle_properties,
    run_bound_kernel,
) -> None:
    bound_kernel = construct_compute_z_kernel(linear_vertical_transform, z="my_z", h="my_h", zeta="my_zeta")

    base = make_particle_properties(zidx=[1.5], yidx=[1.5], xidx=[1.5], z=[np.nan])
    particle_properties = {
        "status": base["status"],
        "zidx": base["zidx"],
        "yidx": base["yidx"],
        "xidx": base["xidx"],
        "my_z": base["z"],
    }
    field_data = {"my_h": uniform_h_zeta_field_data["h"], "my_zeta": uniform_h_zeta_field_data["zeta"]}
    run_bound_kernel(bound_kernel, particle_properties, {}, field_data)

    expected = make_particle_properties(zidx=[1.5], yidx=[1.5], xidx=[1.5], z=[np.nan])
    linear_vertical_transform.compute_z_kernel()(expected, {}, uniform_h_zeta_field_data)

    assert particle_properties["my_z"][0] == pytest.approx(expected["z"][0])


def test_construct_compute_zidx_kernel_runs_correctly_with_custom_bindings(
    linear_vertical_transform: LinearVerticalTransform2,
    uniform_h_zeta_field_data,
    make_particle_properties,
    run_bound_kernel,
) -> None:
    bound_kernel = construct_compute_zidx_kernel(linear_vertical_transform, z="my_z", h="my_h", zeta="my_zeta")

    base = make_particle_properties(zidx=[np.nan], yidx=[1.5], xidx=[1.5], z=[-20.0])
    particle_properties = {
        "status": base["status"],
        "zidx": base["zidx"],
        "yidx": base["yidx"],
        "xidx": base["xidx"],
        "my_z": base["z"],
    }
    field_data = {"my_h": uniform_h_zeta_field_data["h"], "my_zeta": uniform_h_zeta_field_data["zeta"]}
    run_bound_kernel(bound_kernel, particle_properties, {}, field_data)

    expected = make_particle_properties(zidx=[np.nan], yidx=[1.5], xidx=[1.5], z=[-20.0])
    linear_vertical_transform.compute_zidx_kernel()(expected, {}, uniform_h_zeta_field_data)

    assert particle_properties["zidx"][0] == pytest.approx(expected["zidx"][0])
