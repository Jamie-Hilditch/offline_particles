"""Tests for particle advection kernel constructors.

These tests focus on the public advection APIs and verify that they:

* accept valid velocity and scaling field layouts in 1D, 2D, and 3D,
* map particle indices correctly through the interpolation wrappers,
* apply the metric versus grid-spacing scaling rule correctly,
* preserve inactive particles,
* bind field and particle names correctly for the public constructor, and
* reject invalid dimensionality, duplicate axes, and invalid stencil sizes.
"""

from collections.abc import Iterable

import numpy as np
import pytest

from offline_particles.fields import FieldData
from offline_particles.kernels._kernels import BoundKernel, ParticleKernel
from offline_particles.kernels.advection import advection_particle_kernel_factory, construct_advection_kernel
from offline_particles.kernels.status import INACTIVE_FLAG
from offline_particles.spatial_arrays import ArrayAxis
from tests.kernels.conftest import PARTICLE_COORDS, offsets_in_layout

_VELOCITY_BASE = 5.0
_SCALING_BASE = 3.0
_VELOCITY_COEFFS = (1.0, 10.0, 100.0)
_SCALING_COEFFS = (2.0, 20.0, 200.0)

_SUPPORTED_CASES = [
    ((ArrayAxis.Z,), (ArrayAxis.X,)),
    ((ArrayAxis.Y,), (ArrayAxis.Z, ArrayAxis.X)),
    ((ArrayAxis.X,), (ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X)),
    ((ArrayAxis.Y, ArrayAxis.X), (ArrayAxis.Z,)),
    ((ArrayAxis.Z, ArrayAxis.Y), (ArrayAxis.X, ArrayAxis.Z)),
    ((ArrayAxis.X, ArrayAxis.Z), (ArrayAxis.Y, ArrayAxis.X, ArrayAxis.Z)),
    ((ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X), (ArrayAxis.Y,)),
    ((ArrayAxis.Y, ArrayAxis.Z, ArrayAxis.X), (ArrayAxis.X, ArrayAxis.Z)),
    ((ArrayAxis.X, ArrayAxis.Y, ArrayAxis.Z), (ArrayAxis.Z, ArrayAxis.X, ArrayAxis.Y)),
]

_REPRESENTATIVE_CASES = [
    _SUPPORTED_CASES[0],
    _SUPPORTED_CASES[4],
    _SUPPORTED_CASES[8],
]


def _field_shape_for_ndim(ndim: int, N: int) -> tuple[int, ...]:
    return tuple(2 * N + 4 + axis_index for axis_index in range(ndim))


def _build_affine_field(
    layout: tuple[ArrayAxis, ...],
    N: int,
    *,
    base: float,
    coefficients: Iterable[float],
) -> tuple[np.ndarray, tuple[float, ...]]:
    shape = _field_shape_for_ndim(len(layout), N)
    grids = np.meshgrid(*(np.arange(size, dtype=np.float64) for size in shape), indexing="ij")
    field = np.full(shape, base, dtype=np.float64)
    for coefficient, grid in zip(tuple(coefficients)[: len(layout)], grids, strict=True):
        field = field + coefficient * grid
    return field, offsets_in_layout(layout)


def _affine_value_at(
    layout: tuple[ArrayAxis, ...],
    coefficients: Iterable[float],
    base: float,
    offsets: tuple[float, ...],
) -> float:
    return base + sum(
        coefficient * (PARTICLE_COORDS[axis] + offset)
        for coefficient, axis, offset in zip(tuple(coefficients)[: len(layout)], layout, offsets, strict=True)
    )


def _kernel_call_dict(bound_kernel: BoundKernel, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {decl_name: values[binding] for decl_name, binding in bound_kernel.particle_property_bindings.items()}


def _field_call_dict(bound_kernel: BoundKernel, values: dict[str, FieldData]) -> dict[str, FieldData]:
    return {decl_name: values[binding] for decl_name, binding in bound_kernel.field_data_bindings.items()}


def _build_particle_properties() -> dict[str, np.ndarray]:
    return {
        "status": np.array([0, INACTIVE_FLAG], dtype=np.uint8),
        "zidx": np.array([PARTICLE_COORDS[ArrayAxis.Z], PARTICLE_COORDS[ArrayAxis.Z]], dtype=np.float64),
        "yidx": np.array([PARTICLE_COORDS[ArrayAxis.Y], PARTICLE_COORDS[ArrayAxis.Y]], dtype=np.float64),
        "xidx": np.array([PARTICLE_COORDS[ArrayAxis.X], PARTICLE_COORDS[ArrayAxis.X]], dtype=np.float64),
        "idx_tendency": np.array([1.5, -2.5], dtype=np.float64),
    }


@pytest.mark.parametrize("velocity_layout,scaling_layout", _SUPPORTED_CASES)
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("metric", [True, False])
def test_advection_particle_kernel_factory_matches_affine_fields(
    velocity_layout: tuple[ArrayAxis, ...],
    scaling_layout: tuple[ArrayAxis, ...],
    N: int,
    metric: bool,
) -> None:
    kernel = advection_particle_kernel_factory(velocity_layout, scaling_layout, N=N, metric=metric)
    assert isinstance(kernel, ParticleKernel)

    velocity_field, velocity_offsets = _build_affine_field(
        velocity_layout,
        N,
        base=_VELOCITY_BASE,
        coefficients=_VELOCITY_COEFFS,
    )
    scaling_field, scaling_offsets = _build_affine_field(
        scaling_layout,
        N,
        base=_SCALING_BASE,
        coefficients=_SCALING_COEFFS,
    )
    particle_properties = _build_particle_properties()
    field_data = {
        "velocity": FieldData(velocity_field, velocity_offsets),
        "scaling": FieldData(scaling_field, scaling_offsets),
    }

    expected_velocity = _affine_value_at(velocity_layout, _VELOCITY_COEFFS, _VELOCITY_BASE, velocity_offsets)
    expected_scaling = _affine_value_at(scaling_layout, _SCALING_COEFFS, _SCALING_BASE, scaling_offsets)
    expected_delta = expected_velocity * expected_scaling if metric else expected_velocity / expected_scaling

    expected = particle_properties["idx_tendency"].copy()
    expected[0] += expected_delta

    kernel(
        particle_properties,
        {},
        field_data,
    )

    np.testing.assert_allclose(particle_properties["idx_tendency"], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(particle_properties["idx_tendency"][1:], np.array([-2.5], dtype=np.float64))


@pytest.mark.parametrize("velocity_layout,scaling_layout", _REPRESENTATIVE_CASES)
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("metric", [True, False])
def test_construct_advection_kernel_binds_and_matches_affine_fields(
    velocity_layout: tuple[ArrayAxis, ...],
    scaling_layout: tuple[ArrayAxis, ...],
    N: int,
    metric: bool,
) -> None:
    bound_kernel = construct_advection_kernel(
        "idx_tendency_out",
        "velocity_input",
        "scaling_input",
        velocity_layout,
        scaling_layout,
        N=N,
        metric=metric,
    )

    assert isinstance(bound_kernel, BoundKernel)
    assert bound_kernel.particle_property_bindings["idx_tendency"] == "idx_tendency_out"
    assert bound_kernel.field_data_bindings["velocity"] == "velocity_input"
    assert bound_kernel.field_data_bindings["scaling"] == "scaling_input"
    assert len(bound_kernel.kernel.field_data["velocity"]._layout_validators) == 1
    assert len(bound_kernel.kernel.field_data["scaling"]._layout_validators) == 1

    velocity_field, velocity_offsets = _build_affine_field(
        velocity_layout,
        N,
        base=_VELOCITY_BASE,
        coefficients=_VELOCITY_COEFFS,
    )
    scaling_field, scaling_offsets = _build_affine_field(
        scaling_layout,
        N,
        base=_SCALING_BASE,
        coefficients=_SCALING_COEFFS,
    )
    particle_properties = _build_particle_properties()
    particle_properties["idx_tendency_out"] = particle_properties.pop("idx_tendency")
    field_data = {
        "velocity_input": FieldData(velocity_field, velocity_offsets),
        "scaling_input": FieldData(scaling_field, scaling_offsets),
    }

    expected_velocity = _affine_value_at(velocity_layout, _VELOCITY_COEFFS, _VELOCITY_BASE, velocity_offsets)
    expected_scaling = _affine_value_at(scaling_layout, _SCALING_COEFFS, _SCALING_BASE, scaling_offsets)
    expected_delta = expected_velocity * expected_scaling if metric else expected_velocity / expected_scaling

    expected = particle_properties["idx_tendency_out"].copy()
    expected[0] += expected_delta

    bound_kernel.kernel(
        _kernel_call_dict(bound_kernel, particle_properties),
        {},
        _field_call_dict(bound_kernel, field_data),
    )

    np.testing.assert_allclose(particle_properties["idx_tendency_out"], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "factory",
    [advection_particle_kernel_factory, construct_advection_kernel],
)
@pytest.mark.parametrize(
    "bad_velocity_layout,bad_scaling_layout,match",
    [
        ((ArrayAxis.X, ArrayAxis.X), (ArrayAxis.Z,), "Duplicate dimensions in dim_ordering"),
        ((ArrayAxis.Z,), (ArrayAxis.Y, ArrayAxis.Y), "Duplicate dimensions in dim_ordering"),
        ((), (ArrayAxis.Z,), "Unsupported number of dimensions"),
    ],
)
def test_advection_constructors_reject_invalid_dim_orderings(
    factory,
    bad_velocity_layout: tuple[ArrayAxis, ...],
    bad_scaling_layout: tuple[ArrayAxis, ...],
    match: str,
) -> None:
    kwargs = {
        "N": 1,
        "metric": True,
    }
    if factory is construct_advection_kernel:
        with pytest.raises(ValueError, match=match):
            factory(
                "idx_tendency_out",
                "velocity_input",
                "scaling_input",
                bad_velocity_layout,
                bad_scaling_layout,
                **kwargs,
            )
    else:
        with pytest.raises(ValueError, match=match):
            factory(bad_velocity_layout, bad_scaling_layout, **kwargs)


@pytest.mark.parametrize("factory", [advection_particle_kernel_factory, construct_advection_kernel])
@pytest.mark.parametrize("invalid_n", [0, -1])
def test_advection_constructors_reject_nonpositive_N(factory, invalid_n: int) -> None:
    with pytest.raises(ValueError, match="N must be a positive integer"):
        if factory is construct_advection_kernel:
            factory(
                "idx_tendency_out",
                "velocity_input",
                "scaling_input",
                (ArrayAxis.Z,),
                (ArrayAxis.X,),
                N=invalid_n,
                metric=True,
            )
        else:
            factory((ArrayAxis.Z,), (ArrayAxis.X,), N=invalid_n, metric=True)


@pytest.mark.parametrize("metric", [True, False])
def test_advection_metric_flag_changes_expected_operation(metric: bool) -> None:
    velocity_layout = (ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X)
    scaling_layout = (ArrayAxis.X, ArrayAxis.Z, ArrayAxis.Y)
    N = 1

    kernel = advection_particle_kernel_factory(velocity_layout, scaling_layout, N=N, metric=metric)
    velocity_field, velocity_offsets = _build_affine_field(
        velocity_layout,
        N,
        base=_VELOCITY_BASE,
        coefficients=_VELOCITY_COEFFS,
    )
    scaling_field, scaling_offsets = _build_affine_field(
        scaling_layout,
        N,
        base=_SCALING_BASE,
        coefficients=_SCALING_COEFFS,
    )
    particle_properties = _build_particle_properties()
    field_data = {
        "velocity": FieldData(velocity_field, velocity_offsets),
        "scaling": FieldData(scaling_field, scaling_offsets),
    }

    expected_velocity = _affine_value_at(velocity_layout, _VELOCITY_COEFFS, _VELOCITY_BASE, velocity_offsets)
    expected_scaling = _affine_value_at(scaling_layout, _SCALING_COEFFS, _SCALING_BASE, scaling_offsets)
    expected_delta = expected_velocity * expected_scaling if metric else expected_velocity / expected_scaling

    kernel(particle_properties, {}, field_data)

    assert particle_properties["idx_tendency"][0] == pytest.approx(1.5 + expected_delta)
