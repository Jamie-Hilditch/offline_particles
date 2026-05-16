"""Tests for linear and quadratic relaxation kernel constructors."""

import re

import numpy as np
import pytest

from offline_particles.fields import FieldData
from offline_particles.kernels._kernels import BoundKernel
from offline_particles.kernels.relaxation import (
    construct_linear_damping_kernel,
    construct_linear_relaxation_kernel,
    construct_quadratic_damping_kernel,
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


def _construct_relaxation_public_kernel(
    form: str,
    *,
    constant_coefficient: np.inexact | float | None = None,
    property_coefficient: str | None = None,
    scalar_coefficient: str | None = None,
    constant_target: np.inexact | float | None = None,
    property_target: str | None = None,
    scalar_target: str | None = None,
    field_target: str | None = None,
    array_layout: ArrayLayout | None = None,
) -> BoundKernel:
    if form == "linear":
        return construct_linear_relaxation_kernel(
            "my_prop",
            "my_dprop",
            constant_coefficient=constant_coefficient,
            property_coefficient=property_coefficient,
            scalar_coefficient=scalar_coefficient,
            constant_target=constant_target,
            property_target=property_target,
            scalar_target=scalar_target,
            field_target=field_target,
            array_layout=array_layout,
        )

    if form == "quadratic":
        return construct_quadratic_relaxation_kernel(
            "my_prop",
            "my_dprop",
            constant_coefficient=constant_coefficient,
            property_coefficient=property_coefficient,
            scalar_coefficient=scalar_coefficient,
            constant_target=constant_target,
            property_target=property_target,
            scalar_target=scalar_target,
            field_target=field_target,
            array_layout=array_layout,
        )

    raise ValueError(f"invalid form={form}")


def _construct_damping_public_kernel(
    form: str,
    *,
    constant_coefficient: np.inexact | float | None = None,
    property_coefficient: str | None = None,
    scalar_coefficient: str | None = None,
) -> BoundKernel:
    if form == "linear":
        return construct_linear_damping_kernel(
            "my_prop",
            "my_dprop",
            constant_coefficient=constant_coefficient,
            property_coefficient=property_coefficient,
            scalar_coefficient=scalar_coefficient,
        )

    if form == "quadratic":
        return construct_quadratic_damping_kernel(
            "my_prop",
            "my_dprop",
            constant_coefficient=constant_coefficient,
            property_coefficient=property_coefficient,
            scalar_coefficient=scalar_coefficient,
        )

    raise ValueError(f"invalid form={form}")


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

    return constructor(**kwargs)  # type: ignore[arg-type]


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
        "my_prop": prop,
        "my_dprop": dprop,
        "my_relaxation_coefficient": relaxation_coefficient_property,
        "my_target": target_property,
        "xidx": xidx,
        "yidx": yidx,
        "zidx": zidx,
    }

    scalars = {
        "my_relaxation_coefficient": np.float64(0.2),
        "my_target": np.float64(1.4),
    }

    field_data = {}
    if target_kind == "field":
        assert field_layout_axes is not None
        shape = (4,) * len(field_layout_axes)
        field_array = np.full(shape, 1.1, dtype=np.float64)
        offsets = (0.0,) * len(field_layout_axes)
        field_data = {"my_target": FieldData(field_array, offsets)}

    if coefficient_kind == "constant":
        coefficient = _CONST_COEFF
    elif coefficient_kind == "property":
        coefficient = float(relaxation_coefficient_property[0])
    elif coefficient_kind == "scalar":
        coefficient = float(scalars["my_relaxation_coefficient"])
    else:
        raise ValueError(f"invalid coefficient_kind={coefficient_kind}")

    if target_kind == "constant":
        target = _CONST_TARGET
    elif target_kind == "property":
        target = float(target_property[0])
    elif target_kind == "scalar":
        target = float(scalars["my_target"])
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

    expected_active = particle_properties["my_dprop"][0] + _expected_increment(
        form=form,
        coefficient=coefficient,
        target=target,
        prop=particle_properties["my_prop"][0],
    )

    kernel_particle_properties = {
        decl_name: particle_properties[binding]
        for decl_name, binding in bound_kernel.particle_property_bindings.items()
    }
    kernel_scalars = {decl_name: scalars[binding] for decl_name, binding in bound_kernel.scalar_bindings.items()}
    kernel_field_data = {
        decl_name: field_data[binding] for decl_name, binding in bound_kernel.field_data_bindings.items()
    }

    bound_kernel.kernel(kernel_particle_properties, kernel_scalars, kernel_field_data)

    assert particle_properties["my_dprop"][0] == pytest.approx(expected_active)
    assert particle_properties["my_dprop"][1] == pytest.approx(-0.8)


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

    expected_active = particle_properties["my_dprop"][0] + _expected_increment(
        form=form,
        coefficient=coefficient,
        target=target,
        prop=particle_properties["my_prop"][0],
    )

    kernel_particle_properties = {
        decl_name: particle_properties[binding]
        for decl_name, binding in bound_kernel.particle_property_bindings.items()
    }
    kernel_scalars = {decl_name: scalars[binding] for decl_name, binding in bound_kernel.scalar_bindings.items()}
    kernel_field_data = {
        decl_name: field_data[binding] for decl_name, binding in bound_kernel.field_data_bindings.items()
    }

    bound_kernel.kernel(kernel_particle_properties, kernel_scalars, kernel_field_data)

    assert particle_properties["my_dprop"][0] == pytest.approx(expected_active)
    assert particle_properties["my_dprop"][1] == pytest.approx(-0.8)


@pytest.mark.parametrize("form", _FORMS)
def test_relaxation_public_api_accepts_only_valid_argument_combinations(form: str) -> None:
    coefficient_arguments: tuple[tuple[str, np.float64 | str], ...] = (
        ("constant_coefficient", np.float64(0.2)),
        ("property_coefficient", "my_relaxation_coefficient"),
        ("scalar_coefficient", "my_relaxation_coefficient"),
    )
    target_arguments: tuple[tuple[str, np.float64 | str], ...] = (
        ("constant_target", np.float64(1.1)),
        ("property_target", "my_target"),
        ("scalar_target", "my_target"),
        ("field_target", "my_target"),
    )

    for coefficient_mask in range(1 << len(coefficient_arguments)):
        selected_coefficient_arguments = [
            coefficient_arguments[i]
            for i in range(len(coefficient_arguments))
            if coefficient_mask & (1 << i)
        ]
        for target_mask in range(1 << len(target_arguments)):
            selected_target_arguments = [
                target_arguments[i]
                for i in range(len(target_arguments))
                if target_mask & (1 << i)
            ]
            has_field_target = any(name == "field_target" for name, _ in selected_target_arguments)
            array_layout_options = (False, True) if has_field_target else (False,)

            for include_array_layout in array_layout_options:
                kwargs = dict(selected_coefficient_arguments)
                kwargs.update(selected_target_arguments)
                if include_array_layout:
                    kwargs["array_layout"] = ArrayLayout(("X",), ("center",))

                valid = (
                    len(selected_coefficient_arguments) == 1
                    and len(selected_target_arguments) == 1
                    and (not has_field_target or include_array_layout)
                )
                if valid:
                    kernel = _construct_relaxation_public_kernel(form, **kwargs)
                    assert isinstance(kernel, BoundKernel)
                else:
                    expected_error_message = (
                        "`array_layout` must be provided when using a field target."
                        if has_field_target and not include_array_layout
                        else (
                            "Exactly one coefficient (constant/property/scalar) and "
                            "one target (constant/property/scalar/field) must be provided."
                        )
                    )
                    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
                        _construct_relaxation_public_kernel(form, **kwargs)


@pytest.mark.parametrize("form", _FORMS)
def test_damping_public_api_accepts_only_valid_argument_combinations(form: str) -> None:
    coefficient_arguments: tuple[tuple[str, np.float64 | str], ...] = (
        ("constant_coefficient", np.float64(0.2)),
        ("property_coefficient", "my_damping_coefficient"),
        ("scalar_coefficient", "my_damping_coefficient"),
    )

    for coefficient_mask in range(1 << len(coefficient_arguments)):
        kwargs = {
            coefficient_arguments[i][0]: coefficient_arguments[i][1]
            for i in range(len(coefficient_arguments))
            if coefficient_mask & (1 << i)
        }
        if len(kwargs) == 1:
            kernel = _construct_damping_public_kernel(form, **kwargs)
            assert isinstance(kernel, BoundKernel)
        else:
            with pytest.raises(
                ValueError,
                match="Exactly one coefficient \\(constant/property/scalar\\) must be provided\\.",
            ):
                _construct_damping_public_kernel(form, **kwargs)
