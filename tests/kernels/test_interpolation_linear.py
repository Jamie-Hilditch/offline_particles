"""Tests for the linear interpolation kernel constructors."""

import numpy as np
import pytest

from offline_particles.kernels._kernels import BoundKernel
from offline_particles.kernels.interpolation import (
    construct_1D_interpolation_kernel,
    construct_2D_interpolation_kernel,
    construct_3D_interpolation_kernel,
    construct_X_interpolation_kernel,
    construct_XY_interpolation_kernel,
    construct_XYZ_interpolation_kernel,
    construct_XZ_interpolation_kernel,
    construct_XZY_interpolation_kernel,
    construct_Y_interpolation_kernel,
    construct_YX_interpolation_kernel,
    construct_YXZ_interpolation_kernel,
    construct_YZ_interpolation_kernel,
    construct_YZX_interpolation_kernel,
    construct_Z_interpolation_kernel,
    construct_ZX_interpolation_kernel,
    construct_ZXY_interpolation_kernel,
    construct_ZY_interpolation_kernel,
    construct_ZYX_interpolation_kernel,
)
from offline_particles.spatial_arrays import ArrayAxis


class TestConstructLinearInterpolationKernel:
    """Tests for construct_linear_interpolation_kernel."""

    def test_returns_bound_kernel(self) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field")
        assert isinstance(kernel, BoundKernel)

    def test_z_axis_binds_zidx(self) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field")
        assert kernel.particle_property_bindings["idx"] == "zidx"

    def test_y_axis_binds_yidx(self) -> None:
        kernel = construct_1D_interpolation_kernel("Y", "temperature", "temp_field")
        assert kernel.particle_property_bindings["idx"] == "yidx"

    def test_x_axis_binds_xidx(self) -> None:
        kernel = construct_1D_interpolation_kernel("X", "temperature", "temp_field")
        assert kernel.particle_property_bindings["idx"] == "xidx"

    def test_array_axis_enum_accepted(self) -> None:
        kernel = construct_1D_interpolation_kernel(ArrayAxis.Z, "temperature", "temp_field")
        assert kernel.particle_property_bindings["idx"] == "zidx"

    def test_output_binding_uses_given_name(self) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "my_output", "my_field")
        assert kernel.particle_property_bindings["output"] == "my_output"

    def test_field_binding_uses_given_name(self) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "my_output", "my_field")
        assert kernel.field_data_bindings["field"] == "my_field"

    def test_default_field_dtype_is_float64(self) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field")
        assert kernel.kernel.field_data["field"].dtype == np.dtype(np.float64)

    def test_custom_field_dtype_is_applied(self) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field", field_dtype=np.float32)
        assert kernel.kernel.field_data["field"].dtype == np.dtype(np.float32)

    def test_output_dtype_defaults_to_field_dtype(self) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field", field_dtype=np.float32)
        assert kernel.kernel.particle_properties["output"].dtype == np.dtype(np.float32)

    def test_custom_output_dtype_is_applied(self) -> None:
        kernel = construct_1D_interpolation_kernel(
            "Z", "temperature", "temp_field", field_dtype=np.float64, output_dtype=np.float32
        )
        assert kernel.kernel.particle_properties["output"].dtype == np.dtype(np.float32)

    def test_invalid_axis_string_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid ArrayAxis"):
            construct_1D_interpolation_kernel("INVALID", "temperature", "temp_field")

    def test_lowercase_axis_string_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid ArrayAxis"):
            construct_1D_interpolation_kernel("z", "temperature", "temp_field")

    def test_field_data_has_layout_validator(self) -> None:
        """The field data declaration should have exactly one layout validator."""
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field")
        field_decl = kernel.kernel.field_data["field"]
        assert len(field_decl._layout_validators) == 1


class TestConstructBilinearInterpolationKernel:
    """Tests for construct_bilinear_interpolation_kernel."""

    def test_returns_bound_kernel(self) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field")
        assert isinstance(kernel, BoundKernel)

    def test_zy_axes_bind_correct_indices(self) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field")
        assert kernel.particle_property_bindings["idx_0"] == "zidx"
        assert kernel.particle_property_bindings["idx_1"] == "yidx"

    def test_yx_axes_bind_correct_indices(self) -> None:
        kernel = construct_2D_interpolation_kernel(("Y", "X"), "temperature", "temp_field")
        assert kernel.particle_property_bindings["idx_0"] == "yidx"
        assert kernel.particle_property_bindings["idx_1"] == "xidx"

    def test_xz_axes_bind_correct_indices(self) -> None:
        kernel = construct_2D_interpolation_kernel(("X", "Z"), "temperature", "temp_field")
        assert kernel.particle_property_bindings["idx_0"] == "xidx"
        assert kernel.particle_property_bindings["idx_1"] == "zidx"

    def test_array_axis_enum_accepted(self) -> None:
        kernel = construct_2D_interpolation_kernel((ArrayAxis.Z, ArrayAxis.Y), "temperature", "temp_field")
        assert kernel.particle_property_bindings["idx_0"] == "zidx"
        assert kernel.particle_property_bindings["idx_1"] == "yidx"

    def test_output_binding_uses_given_name(self) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "my_output", "my_field")
        assert kernel.particle_property_bindings["output"] == "my_output"

    def test_field_binding_uses_given_name(self) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "my_output", "my_field")
        assert kernel.field_data_bindings["field"] == "my_field"

    def test_default_field_dtype_is_float64(self) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field")
        assert kernel.kernel.field_data["field"].dtype == np.dtype(np.float64)

    def test_custom_field_dtype_is_applied(self) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", field_dtype=np.float32)
        assert kernel.kernel.field_data["field"].dtype == np.dtype(np.float32)

    def test_output_dtype_defaults_to_field_dtype(self) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", field_dtype=np.float32)
        assert kernel.kernel.particle_properties["output"].dtype == np.dtype(np.float32)

    def test_custom_output_dtype_is_applied(self) -> None:
        kernel = construct_2D_interpolation_kernel(
            ("Z", "Y"), "temperature", "temp_field", field_dtype=np.float64, output_dtype=np.float32
        )
        assert kernel.kernel.particle_properties["output"].dtype == np.dtype(np.float32)

    def test_wrong_number_of_axes_raises(self) -> None:
        with pytest.raises(ValueError, match="axes must be a tuple of two elements"):
            construct_2D_interpolation_kernel(("Z",), "temperature", "temp_field")  # type: ignore[call-arg]

    def test_duplicate_axes_raises(self) -> None:
        with pytest.raises(ValueError, match="two axes must be different"):
            construct_2D_interpolation_kernel(("Z", "Z"), "temperature", "temp_field")

    def test_invalid_axis_string_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid ArrayAxis"):
            construct_2D_interpolation_kernel(("INVALID", "Y"), "temperature", "temp_field")

    def test_field_data_has_layout_validator(self) -> None:
        """The field data declaration should have exactly one layout validator."""
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field")
        field_decl = kernel.kernel.field_data["field"]
        assert len(field_decl._layout_validators) == 1


class TestConstructTrilinearInterpolationKernel:
    """Tests for construct_trilinear_interpolation_kernel."""

    def test_returns_bound_kernel(self) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field")
        assert isinstance(kernel, BoundKernel)

    def test_zyx_axes_bind_correct_indices(self) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field")
        assert kernel.particle_property_bindings["idx_0"] == "zidx"
        assert kernel.particle_property_bindings["idx_1"] == "yidx"
        assert kernel.particle_property_bindings["idx_2"] == "xidx"

    def test_xyz_axes_bind_correct_indices(self) -> None:
        kernel = construct_3D_interpolation_kernel(("X", "Y", "Z"), "temperature", "temp_field")
        assert kernel.particle_property_bindings["idx_0"] == "xidx"
        assert kernel.particle_property_bindings["idx_1"] == "yidx"
        assert kernel.particle_property_bindings["idx_2"] == "zidx"

    def test_xzy_axes_bind_correct_indices(self) -> None:
        kernel = construct_3D_interpolation_kernel(("X", "Z", "Y"), "temperature", "temp_field")
        assert kernel.particle_property_bindings["idx_0"] == "xidx"
        assert kernel.particle_property_bindings["idx_1"] == "zidx"
        assert kernel.particle_property_bindings["idx_2"] == "yidx"

    def test_array_axis_enum_accepted(self) -> None:
        kernel = construct_3D_interpolation_kernel((ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X), "temperature", "temp_field")
        assert kernel.particle_property_bindings["idx_0"] == "zidx"
        assert kernel.particle_property_bindings["idx_1"] == "yidx"
        assert kernel.particle_property_bindings["idx_2"] == "xidx"

    def test_output_binding_uses_given_name(self) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "my_output", "my_field")
        assert kernel.particle_property_bindings["output"] == "my_output"

    def test_field_binding_uses_given_name(self) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "my_output", "my_field")
        assert kernel.field_data_bindings["field"] == "my_field"

    def test_default_field_dtype_is_float64(self) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field")
        assert kernel.kernel.field_data["field"].dtype == np.dtype(np.float64)

    def test_custom_field_dtype_is_applied(self) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", field_dtype=np.float32)
        assert kernel.kernel.field_data["field"].dtype == np.dtype(np.float32)

    def test_output_dtype_defaults_to_field_dtype(self) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", field_dtype=np.float32)
        assert kernel.kernel.particle_properties["output"].dtype == np.dtype(np.float32)

    def test_custom_output_dtype_is_applied(self) -> None:
        kernel = construct_3D_interpolation_kernel(
            ("Z", "Y", "X"), "temperature", "temp_field", field_dtype=np.float64, output_dtype=np.float32
        )
        assert kernel.kernel.particle_properties["output"].dtype == np.dtype(np.float32)

    def test_wrong_number_of_axes_raises(self) -> None:
        with pytest.raises(ValueError, match="axes must be a tuple of three elements"):
            construct_3D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field")  # type: ignore[arg-type]

    def test_duplicate_axes_raises(self) -> None:
        with pytest.raises(ValueError, match="All three axes must be different"):
            construct_3D_interpolation_kernel(("Z", "Z", "X"), "temperature", "temp_field")

    def test_invalid_axis_string_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid ArrayAxis"):
            construct_3D_interpolation_kernel(("INVALID", "Y", "X"), "temperature", "temp_field")

    def test_field_data_has_layout_validator(self) -> None:
        """The field data declaration should have exactly one layout validator."""
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field")
        field_decl = kernel.kernel.field_data["field"]
        assert len(field_decl._layout_validators) == 1


class TestConvenienceConstructors:
    """Smoke tests for the partially-applied convenience constructors."""

    @pytest.mark.parametrize(
        "constructor,expected_bindings",
        [
            (construct_Z_interpolation_kernel, {"idx": "zidx"}),
            (construct_Y_interpolation_kernel, {"idx": "yidx"}),
            (construct_X_interpolation_kernel, {"idx": "xidx"}),
        ],
    )
    def test_1d_constructors_bind_correct_index(self, constructor, expected_bindings) -> None:
        kernel = constructor(output="temperature", field="temp_field")
        assert isinstance(kernel, BoundKernel)
        for prop, expected in expected_bindings.items():
            assert kernel.particle_property_bindings[prop] == expected

    @pytest.mark.parametrize(
        "constructor,expected_idx_0,expected_idx_1",
        [
            (construct_XY_interpolation_kernel, "xidx", "yidx"),
            (construct_XZ_interpolation_kernel, "xidx", "zidx"),
            (construct_YX_interpolation_kernel, "yidx", "xidx"),
            (construct_YZ_interpolation_kernel, "yidx", "zidx"),
            (construct_ZX_interpolation_kernel, "zidx", "xidx"),
            (construct_ZY_interpolation_kernel, "zidx", "yidx"),
        ],
    )
    def test_2d_constructors_bind_correct_indices(self, constructor, expected_idx_0, expected_idx_1) -> None:
        kernel = constructor(output="temperature", field="temp_field")
        assert isinstance(kernel, BoundKernel)
        assert kernel.particle_property_bindings["idx_0"] == expected_idx_0
        assert kernel.particle_property_bindings["idx_1"] == expected_idx_1

    @pytest.mark.parametrize(
        "constructor,expected_idx_0,expected_idx_1,expected_idx_2",
        [
            (construct_XYZ_interpolation_kernel, "xidx", "yidx", "zidx"),
            (construct_XZY_interpolation_kernel, "xidx", "zidx", "yidx"),
            (construct_YXZ_interpolation_kernel, "yidx", "xidx", "zidx"),
            (construct_YZX_interpolation_kernel, "yidx", "zidx", "xidx"),
            (construct_ZXY_interpolation_kernel, "zidx", "xidx", "yidx"),
            (construct_ZYX_interpolation_kernel, "zidx", "yidx", "xidx"),
        ],
    )
    def test_3d_constructors_bind_correct_indices(
        self, constructor, expected_idx_0, expected_idx_1, expected_idx_2
    ) -> None:
        kernel = constructor(output="temperature", field="temp_field")
        assert isinstance(kernel, BoundKernel)
        assert kernel.particle_property_bindings["idx_0"] == expected_idx_0
        assert kernel.particle_property_bindings["idx_1"] == expected_idx_1
        assert kernel.particle_property_bindings["idx_2"] == expected_idx_2
