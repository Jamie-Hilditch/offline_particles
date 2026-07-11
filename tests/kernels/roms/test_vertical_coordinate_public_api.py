"""Tests for the `construct_compute_z_kernel`/`construct_compute_zidx_kernel` public API.

These check the *wiring* (declarations, default and custom bindings) rather than the math, which
is covered by `test_compute_z_kernel.py`/`test_compute_zidx_kernel.py`.
"""

import pytest

from offline_particles.kernels._kernels import BoundKernel
from offline_particles.kernels.roms import construct_compute_z_kernel, construct_compute_zidx_kernel

_CONSTRUCTORS = (construct_compute_z_kernel, construct_compute_zidx_kernel)


@pytest.mark.parametrize("constructor", _CONSTRUCTORS)
def test_constructor_uses_declared_names_as_default_bindings(constructor, hc_nz: tuple[float, int]) -> None:
    hc, NZ = hc_nz
    bound_kernel = constructor(hc=hc, NZ=NZ)

    assert isinstance(bound_kernel, BoundKernel)
    for declared_name, bound_name in bound_kernel.particle_property_bindings.items():
        assert bound_name == declared_name
    for declared_name, bound_name in bound_kernel.scalar_bindings.items():
        assert bound_name == declared_name
    for declared_name, bound_name in bound_kernel.field_data_bindings.items():
        assert bound_name == declared_name


@pytest.mark.parametrize("constructor", _CONSTRUCTORS)
def test_constructor_honours_custom_bindings(constructor, hc_nz: tuple[float, int]) -> None:
    hc, NZ = hc_nz
    bound_kernel = constructor(hc=hc, NZ=NZ, z="my_z", h="my_h", zeta="my_zeta", C="my_C")

    assert bound_kernel.particle_property_bindings["z"] == "my_z"
    assert bound_kernel.field_data_bindings["h"] == "my_h"
    assert bound_kernel.field_data_bindings["zeta"] == "my_zeta"
    assert bound_kernel.field_data_bindings["C"] == "my_C"


@pytest.mark.parametrize("constructor", _CONSTRUCTORS)
def test_constructor_declares_the_full_shared_input_set(constructor, hc_nz: tuple[float, int]) -> None:
    hc, NZ = hc_nz
    bound_kernel = constructor(hc=hc, NZ=NZ)

    assert set(bound_kernel.kernel.particle_properties) == {"status", "zidx", "yidx", "xidx", "z"}
    assert set(bound_kernel.kernel.scalars) == set()
    assert set(bound_kernel.kernel.field_data) == {"h", "zeta", "C"}


def test_both_constructors_declare_identical_inputs(hc_nz: tuple[float, int]) -> None:
    """`roms_ab3_timestepper` shares bindings between the two kernels, relying on this implicitly."""
    hc, NZ = hc_nz
    z_kernel = construct_compute_z_kernel(hc=hc, NZ=NZ).kernel
    zidx_kernel = construct_compute_zidx_kernel(hc=hc, NZ=NZ).kernel

    assert set(z_kernel.particle_properties) == set(zidx_kernel.particle_properties)
    assert set(z_kernel.scalars) == set(zidx_kernel.scalars)
    assert set(z_kernel.field_data) == set(zidx_kernel.field_data)
