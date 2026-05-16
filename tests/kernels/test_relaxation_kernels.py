"""Tests for linear and quadratic relaxation kernel constructors."""

import numpy as np
import pytest

from offline_particles.fields import FieldData
from offline_particles.kernels.relaxation import (
    construct_linear_relaxation_kernel,
    construct_quadratic_relaxation_kernel,
)
from offline_particles.kernels.status import INACTIVE_FLAG
from offline_particles.spatial_arrays import ArrayLayout

_FORMS = ("linear", "quadratic")
_COEFFICIENT_KINDS = ("constant", "property", "scalar")
_TARGET_KINDS = ("constant", "property", "scalar")
_FIELD_LAYOUTS = (
    ("X",),
    ("Y", "X"),
    ("Z", "Y", "X"),
)

_CONST_COEFF = 0.3
_CONST_TARGET = 1.2


def _construct_relaxation_kernel(
    form: str,
    coefficient_kind: str,
    target_kind: str,
    *,
    field_layout_axes: tuple[str, ...] | None = None,
):
    constructor = construct_linear_relaxation_kernel if form == "linear" else construct_quadratic_relaxation_kernel
    kwargs: dict[str, object] = {"prop": "my_prop", "dprop": "my_dprop", "dtype": np.float64}

    if coefficient_kind == "constant":
        kwargs["constant_coefficient"] = _CONST_COEFF
    elif coefficient_kind == "property":
        kwargs["property_coefficient"] = "my_relaxation_coefficient"
    elif coefficient_kind == "scalar":
        kwargs["scalar_coefficient"] = "my_relaxation_coefficient"
    else:
        raise ValueError(f"invalid coefficient_kind={coefficient_kind}")

    if target_kind == "constant":
        kwargs["constant_target"] = _CONST_TARGET
    elif target_kind == "property":
        kwargs["property_target"] = "my_target"
    elif target_kind == "scalar":
        kwargs["scalar_target"] = "my_target"
    elif target_kind == "field":
        assert field_layout_axes is not None
        kwargs["field_target"] = "my_target"
        kwargs["array_layout"] = ArrayLayout(field_layout_axes, ("center",) * len(field_layout_axes))
    else:
        raise ValueError(f"invalid target_kind={target_kind}")

    return constructor(**kwargs)


def _expected_increment(form: str, coefficient: float, target: float, prop: float) -> float:
    diff = prop - target
    if form == "linear":
        return -coefficient * diff
    if form == "quadratic":
        return -coefficient * diff * abs(diff)
    raise ValueError(f"invalid form={form}")


def _build_kernel_inputs(coefficient_kind: str, target_kind: str, field_layout_axes: tuple[str, ...] | None):
    status = np.array([0, INACTIVE_FLAG], dtype=np.uint8)
    prop = np.array([2.0, -5.0], dtype=np.float64)
    dprop = np.array([0.4, -0.8], dtype=np.float64)
    relaxation_coefficient_property = np.array([0.25, 99.0], dtype=np.float64)
    target_property = np.array([1.5, 999.0], dtype=np.float64)
    xidx = np.array([1.5, 1.5], dtype=np.float64)
    yidx = np.array([1.5, 1.5], dtype=np.float64)
    zidx = np.array([1.5, 1.5], dtype=np.float64)

    particle_properties = {
        "status": status,
        "prop": prop,
        "dprop": dprop,
        "relaxation_coefficient": relaxation_coefficient_property,
        "target": target_property,
        "xidx": xidx,
        "yidx": yidx,
        "zidx": zidx,
    }

    scalars = {
        "relaxation_coefficient": np.float64(0.2),
        "target": np.float64(1.4),
    }

    field_data = {}
    if target_kind == "field":
        assert field_layout_axes is not None
        shape = (4,) * len(field_layout_axes)
        field_array = np.full(shape, 1.1, dtype=np.float64)
        offsets = (0.0,) * len(field_layout_axes)
        field_data = {"target": FieldData(field_array, offsets)}

    if coefficient_kind == "constant":
        coefficient = _CONST_COEFF
    elif coefficient_kind == "property":
        coefficient = float(relaxation_coefficient_property[0])
    elif coefficient_kind == "scalar":
        coefficient = float(scalars["relaxation_coefficient"])
    else:
        raise ValueError(f"invalid coefficient_kind={coefficient_kind}")

    if target_kind == "constant":
        target = _CONST_TARGET
    elif target_kind == "property":
        target = float(target_property[0])
    elif target_kind == "scalar":
        target = float(scalars["target"])
    elif target_kind == "field":
        target = 1.1
    else:
        raise ValueError(f"invalid target_kind={target_kind}")

    return particle_properties, scalars, field_data, coefficient, target


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("coefficient_kind", _COEFFICIENT_KINDS)
@pytest.mark.parametrize("target_kind", _TARGET_KINDS)
def test_relaxation_kernels_cover_all_nonfield_combinations(
    form: str,
    coefficient_kind: str,
    target_kind: str,
) -> None:
    bound_kernel = _construct_relaxation_kernel(form, coefficient_kind, target_kind)
    particle_properties, scalars, field_data, coefficient, target = _build_kernel_inputs(
        coefficient_kind, target_kind, None
    )

    expected_active = particle_properties["dprop"][0] + _expected_increment(
        form=form,
        coefficient=coefficient,
        target=target,
        prop=particle_properties["prop"][0],
    )

    bound_kernel.kernel(particle_properties, scalars, field_data)

    assert particle_properties["dprop"][0] == pytest.approx(expected_active)
    assert particle_properties["dprop"][1] == pytest.approx(-0.8)


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("coefficient_kind", _COEFFICIENT_KINDS)
@pytest.mark.parametrize("field_layout_axes", _FIELD_LAYOUTS)
def test_relaxation_kernels_cover_all_field_target_combinations(
    form: str,
    coefficient_kind: str,
    field_layout_axes: tuple[str, ...],
) -> None:
    bound_kernel = _construct_relaxation_kernel(
        form,
        coefficient_kind,
        "field",
        field_layout_axes=field_layout_axes,
    )
    particle_properties, scalars, field_data, coefficient, target = _build_kernel_inputs(
        coefficient_kind, "field", field_layout_axes
    )

    expected_active = particle_properties["dprop"][0] + _expected_increment(
        form=form,
        coefficient=coefficient,
        target=target,
        prop=particle_properties["prop"][0],
    )

    bound_kernel.kernel(particle_properties, scalars, field_data)

    assert particle_properties["dprop"][0] == pytest.approx(expected_active)
    assert particle_properties["dprop"][1] == pytest.approx(-0.8)
