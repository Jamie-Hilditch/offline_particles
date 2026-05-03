"""Tests for linearly_interpolate_fields."""

import numpy as np
import pytest

from offline_particles.fieldset import Fieldset
from offline_particles.fields import StaticField
from offline_particles.output import linearly_interpolate_fields


def _make_fieldset() -> Fieldset:
    """Create a Fieldset with 1D, 2D, and 3D fields for testing."""
    field_1d = StaticField.from_numpy(
        np.ones((5,), dtype=np.float32),
        axes=("X",),
        staggers=("center",),
    )
    field_2d = StaticField.from_numpy(
        np.ones((4, 5), dtype=np.float32),
        axes=("Y", "X"),
        staggers=("center", "center"),
    )
    field_3d = StaticField.from_numpy(
        np.ones((3, 4, 5), dtype=np.float32),
        axes=("Z", "Y", "X"),
        staggers=("center", "center", "center"),
    )
    return Fieldset(
        10,
        3,
        4,
        5,
        fields={"u": field_1d, "v": field_2d, "w": field_3d},
    )


class TestLinearlyInterpolateFields:
    def test_returns_output_for_each_variable(self) -> None:
        fs = _make_fieldset()
        outputs = linearly_interpolate_fields(fs, "u", "v", "w")
        assert set(outputs.keys()) == {"u", "v", "w"}

    def test_output_key_matches_variable_name(self) -> None:
        fs = _make_fieldset()
        outputs = linearly_interpolate_fields(fs, "u")
        assert "u" in outputs

    def test_1d_field_produces_output_with_kernel(self) -> None:
        fs = _make_fieldset()
        outputs = linearly_interpolate_fields(fs, "u")
        assert len(outputs["u"].kernels) == 1

    def test_2d_field_produces_output_with_kernel(self) -> None:
        fs = _make_fieldset()
        outputs = linearly_interpolate_fields(fs, "v")
        assert len(outputs["v"].kernels) == 1

    def test_3d_field_produces_output_with_kernel(self) -> None:
        fs = _make_fieldset()
        outputs = linearly_interpolate_fields(fs, "w")
        assert len(outputs["w"].kernels) == 1

    def test_missing_variable_raises(self) -> None:
        fs = _make_fieldset()
        with pytest.raises(KeyError, match="missing"):
            linearly_interpolate_fields(fs, "missing")

    def test_custom_particle_property_prefix(self) -> None:
        fs = _make_fieldset()
        outputs = linearly_interpolate_fields(fs, "u", particle_property_prefix="_custom")
        assert outputs["u"].particle_property.name.startswith("_custom")

    def test_default_particle_property_prefix(self) -> None:
        fs = _make_fieldset()
        outputs = linearly_interpolate_fields(fs, "u")
        assert outputs["u"].particle_property.name.startswith("_output")

    def test_no_variables_returns_empty_dict(self) -> None:
        fs = _make_fieldset()
        outputs = linearly_interpolate_fields(fs)
        assert outputs == {}
