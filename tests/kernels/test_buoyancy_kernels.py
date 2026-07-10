"""Tests for the buoyancy force kernel constructor."""

import numpy as np
import pytest

from offline_particles.fields import FieldData
from offline_particles.kernels._kernels import BoundKernel
from offline_particles.kernels.buoyancy import construct_buoyancy_force_kernel
from offline_particles.kernels.status import INACTIVE_FLAG
from offline_particles.spatial_arrays import ArrayLayout

_COEFFICIENT_KINDS = ("constant", "property", "scalar")
_FIELD_LAYOUTS = (
    ("X",),
    ("Y", "X"),
    ("Z", "Y", "X"),
)

_CONST_COEFF = 0.3


def _construct_buoyancy_kernel(coefficient_kind: str, field_layout_axes: tuple[str, ...]) -> BoundKernel:
    kwargs: dict[str, object] = {
        "rhs": "my_tendency",
        "particle_density": "my_prop",
        "density_field": "my_target",
        "array_layout": ArrayLayout(field_layout_axes, ("center",) * len(field_layout_axes)),
    }

    if coefficient_kind == "constant":
        kwargs["constant_coefficient"] = _CONST_COEFF
    elif coefficient_kind == "property":
        kwargs["property_coefficient"] = "my_coefficient"
    elif coefficient_kind == "scalar":
        kwargs["scalar_coefficient"] = "my_coefficient"
    else:
        raise ValueError(f"invalid coefficient_kind={coefficient_kind}")

    return construct_buoyancy_force_kernel(**kwargs)  # type: ignore[arg-type]


def _build_kernel_inputs(coefficient_kind: str, field_layout_axes: tuple[str, ...]):
    status = np.array([0, INACTIVE_FLAG], dtype=np.uint8)
    prop = np.array([2.0, -5.0], dtype=np.float64)
    tendency = np.array([0.4, -0.8], dtype=np.float64)
    coefficient_property = np.array([0.25, 99.0], dtype=np.float64)
    xidx = np.array([1.5, 1.5], dtype=np.float64)
    yidx = np.array([1.5, 1.5], dtype=np.float64)
    zidx = np.array([1.5, 1.5], dtype=np.float64)

    particle_properties = {
        "status": status,
        "my_prop": prop,
        "my_tendency": tendency,
        "my_coefficient": coefficient_property,
        "xidx": xidx,
        "yidx": yidx,
        "zidx": zidx,
    }

    scalars = {
        "my_coefficient": np.float64(0.2),
    }

    shape = (4,) * len(field_layout_axes)
    field_array = np.full(shape, 1.1, dtype=np.float64)
    offsets = (0.0,) * len(field_layout_axes)
    field_data = {"my_target": FieldData(field_array, offsets)}

    if coefficient_kind == "constant":
        coefficient = _CONST_COEFF
    elif coefficient_kind == "property":
        coefficient = float(coefficient_property[0])
    elif coefficient_kind == "scalar":
        coefficient = float(scalars["my_coefficient"])
    else:
        raise ValueError(f"invalid coefficient_kind={coefficient_kind}")

    ambient_density = 1.1

    return particle_properties, scalars, field_data, coefficient, ambient_density


def _run_bound_kernel(bound_kernel: BoundKernel, particle_properties, scalars, field_data) -> None:
    kernel_particle_properties = {
        decl_name: particle_properties[binding]
        for decl_name, binding in bound_kernel.particle_property_bindings.items()
    }
    kernel_scalars = {decl_name: scalars[binding] for decl_name, binding in bound_kernel.scalar_bindings.items()}
    kernel_field_data = {
        decl_name: field_data[binding] for decl_name, binding in bound_kernel.field_data_bindings.items()
    }

    bound_kernel.kernel(kernel_particle_properties, kernel_scalars, kernel_field_data)


@pytest.mark.parametrize("coefficient_kind", _COEFFICIENT_KINDS)
@pytest.mark.parametrize("field_layout_axes", _FIELD_LAYOUTS)
def test_buoyancy_force_kernel_covers_all_dimensionalities_and_coefficient_kinds(
    coefficient_kind: str,
    field_layout_axes: tuple[str, ...],
) -> None:
    bound_kernel = _construct_buoyancy_kernel(coefficient_kind, field_layout_axes)
    particle_properties, scalars, field_data, coefficient, ambient_density = _build_kernel_inputs(
        coefficient_kind, field_layout_axes
    )

    particle_density = particle_properties["my_prop"][0]
    expected_active = particle_properties["my_tendency"][0] + coefficient * (ambient_density - particle_density)

    _run_bound_kernel(bound_kernel, particle_properties, scalars, field_data)

    assert particle_properties["my_tendency"][0] == pytest.approx(expected_active)
    # inactive particle must be left untouched
    assert particle_properties["my_tendency"][1] == pytest.approx(-0.8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"constant_coefficient": _CONST_COEFF, "property_coefficient": "my_coefficient"},
        {"constant_coefficient": _CONST_COEFF, "scalar_coefficient": "my_coefficient"},
        {"property_coefficient": "my_coefficient", "scalar_coefficient": "my_coefficient_scalar"},
        {
            "constant_coefficient": _CONST_COEFF,
            "property_coefficient": "my_coefficient",
            "scalar_coefficient": "my_coefficient_scalar",
        },
    ],
)
def test_buoyancy_force_kernel_rejects_invalid_coefficient_combinations(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Exactly one coefficient \\(constant/property/scalar\\) must be provided\\."):
        construct_buoyancy_force_kernel(
            "my_tendency",
            "my_prop",
            "my_target",
            ArrayLayout(("Z", "Y", "X"), ("center", "center", "center")),
            **kwargs,  # type: ignore[arg-type]
        )


def test_buoyancy_force_kernel_accepts_single_coefficient() -> None:
    kernel = construct_buoyancy_force_kernel(
        "my_tendency",
        "my_prop",
        "my_target",
        ArrayLayout(("Z", "Y", "X"), ("center", "center", "center")),
        constant_coefficient=_CONST_COEFF,
    )
    assert isinstance(kernel, BoundKernel)
