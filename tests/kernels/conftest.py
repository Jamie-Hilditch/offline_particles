"""Shared test support for tests/kernels."""

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
