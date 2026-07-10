"""Shared test support for tests/kernels."""

import pytest

from offline_particles.spatial_arrays import ArrayAxis

PARTICLE_COORDS = {
    ArrayAxis.Z: 1.25,
    ArrayAxis.Y: 2.5,
    ArrayAxis.X: 3.75,
}

FIELD_OFFSETS = {
    ArrayAxis.Z: 0.125,
    ArrayAxis.Y: -0.25,
    ArrayAxis.X: 0.5,
}


def offsets_in_layout(layout: tuple[ArrayAxis, ...]) -> tuple[float, ...]:
    return tuple(FIELD_OFFSETS[axis] for axis in layout)


@pytest.fixture
def run_bound_kernel():
    def _run_bound_kernel(bound_kernel, particle_properties, scalars=None, field_data=None) -> None:
        scalars = scalars or {}
        field_data = field_data or {}
        kernel_particle_properties = {
            decl_name: particle_properties[binding]
            for decl_name, binding in bound_kernel.particle_property_bindings.items()
        }
        kernel_scalars = {decl_name: scalars[binding] for decl_name, binding in bound_kernel.scalar_bindings.items()}
        kernel_field_data = {
            decl_name: field_data[binding] for decl_name, binding in bound_kernel.field_data_bindings.items()
        }
        bound_kernel.kernel(kernel_particle_properties, kernel_scalars, kernel_field_data)

    return _run_bound_kernel
