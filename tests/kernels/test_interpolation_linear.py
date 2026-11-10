"""Tests for the linear interpolation kernel constructors."""

import numpy as np
import pytest

from offline_particles.fields import FieldData
from offline_particles.kernels._kernels import BoundKernel
from offline_particles.kernels.interpolation import (
    _kernel_constructors,
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
    lagrange_1D_particle_factory,
    lagrange_2D_particle_factory,
    lagrange_3D_particle_factory,
)
from offline_particles.spatial_arrays import ArrayAxis


def _stencil_origin_and_local_position(idx: float, N: int, offset: float = 0.0) -> tuple[int, float]:
    offset_idx = idx + offset - (N - 1)
    I0 = int(offset_idx)
    x0 = (N - 1) + offset_idx - I0
    return I0, x0


class TestConstructLinearInterpolationKernel:
    """Tests for construct_1D_interpolation_kernel."""

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
        assert kernel.kernel.field_data["field"].dtype_constraints == (np.float64,)

    def test_custom_field_dtype_is_applied(self) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field", field_dtype=np.float32)
        assert kernel.kernel.field_data["field"].dtype_constraints == (np.float32,)

    def test_output_dtype_defaults_to_field_dtype(self) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field", field_dtype=np.float32)
        assert kernel.kernel.particle_properties["output"].dtype_constraints == (np.float32,)

    def test_custom_output_dtype_is_applied(self) -> None:
        kernel = construct_1D_interpolation_kernel(
            "Z", "temperature", "temp_field", field_dtype=np.float64, output_dtype=np.float32
        )
        assert kernel.kernel.particle_properties["output"].dtype_constraints == (np.float32,)

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

    def test_passes_N_D_and_accumulate_to_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {}

        def fake_factory(N: int, D: int = 0, accumulate: bool = False):
            called["N"] = N
            called["D"] = D
            called["accumulate"] = accumulate

            def kernel_function_impl(*args):
                return None

            return kernel_function_impl

        monkeypatch.setattr(_kernel_constructors, "_lagrange_1D_kernel_function_factory", fake_factory)

        construct_1D_interpolation_kernel("Z", "temperature", "temp_field", N=4, D=1, accumulate=True)
        assert called == {"N": 4, "D": 1, "accumulate": True}

    def test_executes_and_interpolates_correctly(self, run_bound_kernel) -> None:
        """End-to-end: actually run the constructed kernel, not just inspect its bindings.

        This is the only test in this file that invokes the kernel through real
        (particle_properties, scalars, field_data) dicts, exercising the `@kernel_function`
        decorator's key-matching against the declared particle property names.
        """
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field", N=1)
        field = np.array([0.0, 10.0, 40.0, 90.0], dtype=np.float64)
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([2.3], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (0.0,))}

        run_bound_kernel(kernel, particle_properties, {}, field_data)

        assert particle_properties["temperature"][0] == pytest.approx(55.0)

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_exact_for_degree_2n_minus_1_polynomial(self, run_bound_kernel, N: int) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field", N=N)
        degree = 2 * N - 1
        polynomial = np.polynomial.Polynomial(np.arange(1, degree + 2, dtype=np.float64))
        idx = 12.375
        I0, x0 = _stencil_origin_and_local_position(idx, N)
        field = polynomial(np.arange(64, dtype=np.float64) - I0)

        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([idx], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (0.0,))}
        run_bound_kernel(kernel, particle_properties, {}, field_data)

        assert particle_properties["temperature"][0] == pytest.approx(polynomial(x0), rel=1e-11, abs=1e-11)

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_not_exact_for_degree_2n_polynomial(self, run_bound_kernel, N: int) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field", N=N)
        degree = 2 * N
        polynomial = np.polynomial.Polynomial([0.0] * degree + [1.0])
        idx = 12.375
        I0, x0 = _stencil_origin_and_local_position(idx, N)
        field = polynomial(np.arange(64, dtype=np.float64) - I0)

        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([idx], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (0.0,))}
        run_bound_kernel(kernel, particle_properties, {}, field_data)

        assert abs(particle_properties["temperature"][0] - polynomial(x0)) > 1e-10

    def test_rejects_field_smaller_than_stencil(self, run_bound_kernel) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field", N=2)
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([0.5], dtype=np.float64),
            "temperature": np.zeros(1, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(np.ones(3, dtype=np.float64), (0.0,))}

        with pytest.raises(ValueError, match="at least 2N points"):
            run_bound_kernel(kernel, particle_properties, {}, field_data)

    def test_accepts_field_matching_stencil_size(self, run_bound_kernel) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field", N=2)
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([0.5], dtype=np.float64),
            "temperature": np.zeros(1, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(np.ones(4, dtype=np.float64), (0.0,))}

        run_bound_kernel(kernel, particle_properties, {}, field_data)

    def test_rejects_nonpositive_N_at_construction(self) -> None:
        for invalid_n in (0, -1):
            with pytest.raises(ValueError, match="N must be a positive integer"):
                construct_1D_interpolation_kernel("Z", "temperature", "temp_field", N=invalid_n)

    def test_rejects_negative_D_at_construction(self) -> None:
        for invalid_d in (-1, -5):
            with pytest.raises(ValueError, match="derivative_order"):
                construct_1D_interpolation_kernel("Z", "temperature", "temp_field", D=invalid_d)

    def test_derivative_order_is_threaded_through(self, run_bound_kernel) -> None:
        """D should reach the underlying kernel function factory."""
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field", N=2, D=1)
        # f(x) = 3 + 2x; df/dx == 2 everywhere, independent of position
        field = 3.0 + 2.0 * np.arange(10, dtype=np.float64)
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([4.5], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (0.0,))}
        run_bound_kernel(kernel, particle_properties, {}, field_data)

        assert particle_properties["temperature"][0] == pytest.approx(2.0)

    def test_matches_particle_factory_directly(self, run_bound_kernel) -> None:
        kernel = construct_1D_interpolation_kernel("Z", "temperature", "temp_field", N=3)
        field = np.linspace(-1.0, 1.0, 64)
        offset = 0.25
        idx_value = 12.375

        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([idx_value], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (offset,))}
        run_bound_kernel(kernel, particle_properties, {}, field_data)

        single_impl = lagrange_1D_particle_factory(3)
        expected = single_impl(field, np.float64(idx_value + offset), field.shape[0] - 6)

        assert particle_properties["temperature"][0] == pytest.approx(expected)


class TestConstructBilinearInterpolationKernel:
    """Tests for construct_2D_interpolation_kernel."""

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
        assert kernel.kernel.field_data["field"].dtype_constraints == (np.float64,)

    def test_custom_field_dtype_is_applied(self) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", field_dtype=np.float32)
        assert kernel.kernel.field_data["field"].dtype_constraints == (np.float32,)

    def test_output_dtype_defaults_to_field_dtype(self) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", field_dtype=np.float32)
        assert kernel.kernel.particle_properties["output"].dtype_constraints == (np.float32,)

    def test_custom_output_dtype_is_applied(self) -> None:
        kernel = construct_2D_interpolation_kernel(
            ("Z", "Y"), "temperature", "temp_field", field_dtype=np.float64, output_dtype=np.float32
        )
        assert kernel.kernel.particle_properties["output"].dtype_constraints == (np.float32,)

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

    def test_passes_N_D_and_accumulate_to_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {}

        def fake_factory(N0: int, N1: int, D0: int = 0, D1: int = 0, accumulate: bool = False):
            called["N0"] = N0
            called["N1"] = N1
            called["D0"] = D0
            called["D1"] = D1
            called["accumulate"] = accumulate

            def kernel_function_impl(*args):
                return None

            return kernel_function_impl

        monkeypatch.setattr(_kernel_constructors, "_lagrange_2D_kernel_function_factory", fake_factory)

        construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", N=(3, 5), D=(1, 0), accumulate=True)
        assert called == {"N0": 3, "N1": 5, "D0": 1, "D1": 0, "accumulate": True}

    def test_executes_and_interpolates_correctly(self, run_bound_kernel) -> None:
        """End-to-end: actually run the constructed kernel against an affine field.

        Exercises the `@kernel_function` decorator's key-matching for the "idx_0"/"idx_1"
        declared particle properties, which a monkeypatched-factory test cannot catch.
        """
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", N=(1, 1))
        shape = (4, 4)
        field = np.fromfunction(lambda i, j: 7.0 + 1.0 * i + 10.0 * j, shape, dtype=np.float64)
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([1.25], dtype=np.float64),
            "yidx": np.array([2.5], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (0.0, 0.0))}

        run_bound_kernel(kernel, particle_properties, {}, field_data)

        expected = 7.0 + 1.0 * 1.25 + 10.0 * 2.5
        assert particle_properties["temperature"][0] == pytest.approx(expected)

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_exact_for_degree_2n_minus_1_polynomial(self, run_bound_kernel, N: int) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", N=(N, N))
        degree = 2 * N - 1
        idx0, idx1 = 14.25, 16.625
        I0, local0 = _stencil_origin_and_local_position(idx0, N)
        I1, local1 = _stencil_origin_and_local_position(idx1, N)

        def polynomial(x0, x1):
            total = 0.0
            for a in range(degree + 1):
                for b in range(degree + 1):
                    if a + b <= degree:
                        total += (a + 1) * (b + 2) * (x0**a) * (x1**b)
            return total

        grid0 = np.arange(48, dtype=np.float64) - I0
        grid1 = np.arange(40, dtype=np.float64) - I1
        mesh0, mesh1 = np.meshgrid(grid0, grid1, indexing="ij")
        field = polynomial(mesh0, mesh1)

        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([idx0], dtype=np.float64),
            "yidx": np.array([idx1], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (0.0, 0.0))}
        run_bound_kernel(kernel, particle_properties, {}, field_data)

        expected = polynomial(local0, local1)
        assert particle_properties["temperature"][0] == pytest.approx(expected, rel=1e-10, abs=1e-10)

    @pytest.mark.parametrize("shape", [(3, 4), (4, 3), (3, 3)])
    def test_rejects_field_smaller_than_stencil(self, run_bound_kernel, shape: tuple[int, int]) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", N=(2, 2))
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([0.5], dtype=np.float64),
            "yidx": np.array([0.5], dtype=np.float64),
            "temperature": np.zeros(1, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(np.ones(shape, dtype=np.float64), (0.0, 0.0))}

        with pytest.raises(ValueError, match=r"\(2 N0\) x \(2 N1\) points"):
            run_bound_kernel(kernel, particle_properties, {}, field_data)

    def test_accepts_field_matching_stencil_size(self, run_bound_kernel) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", N=(2, 2))
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([0.5], dtype=np.float64),
            "yidx": np.array([0.5], dtype=np.float64),
            "temperature": np.zeros(1, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(np.ones((4, 4), dtype=np.float64), (0.0, 0.0))}

        run_bound_kernel(kernel, particle_properties, {}, field_data)

    def test_accepts_field_with_different_size_per_axis(self, run_bound_kernel) -> None:
        """N is now per-axis, so the minimum field size is no longer symmetric."""
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", N=(1, 3))
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([0.5], dtype=np.float64),
            "yidx": np.array([2.5], dtype=np.float64),
            "temperature": np.zeros(1, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(np.ones((2, 6), dtype=np.float64), (0.0, 0.0))}

        run_bound_kernel(kernel, particle_properties, {}, field_data)

    @pytest.mark.parametrize(("N0", "N1"), [(1, 3), (3, 1)])
    def test_axis_0_exactness_depends_only_on_N0(self, run_bound_kernel, N0: int, N1: int) -> None:
        """A field that only varies along axis 0 isolates axis 0's own stencil capacity.

        Parametrizing both orderings catches an axis mixup the same way the particle-factory-level
        test in test_interpolation_lagrange.py does.
        """
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", N=(N0, N1))
        degree = 2 * N0 - 1
        polynomial = np.polynomial.Polynomial(np.arange(1, degree + 2, dtype=np.float64))
        idx0 = 14.25
        I0, local0 = _stencil_origin_and_local_position(idx0, N0)
        n0, n1 = 2 * N0, 2 * N1
        field = np.tile(polynomial(np.arange(n0 + 20, dtype=np.float64) - I0)[:, None], (1, n1 + 20))

        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([idx0], dtype=np.float64),
            "yidx": np.array([16.625], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (0.0, 0.0))}
        run_bound_kernel(kernel, particle_properties, {}, field_data)

        assert particle_properties["temperature"][0] == pytest.approx(polynomial(local0), rel=1e-9, abs=1e-9)

    def test_rejects_nonpositive_N_on_either_axis_at_construction(self) -> None:
        for invalid_n in (0, -1):
            for N in ((invalid_n, 2), (2, invalid_n)):
                with pytest.raises(ValueError, match="N must be a positive integer"):
                    construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", N=N)

    def test_rejects_negative_D_on_either_axis_at_construction(self) -> None:
        for D in ((-1, 0), (0, -1)):
            with pytest.raises(ValueError, match="derivative_order"):
                construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", D=D)

    def test_rejects_wrong_length_N_at_construction(self) -> None:
        for invalid_n in [(1,), (1, 1, 1)]:
            with pytest.raises(ValueError, match="N must be a tuple of two integers"):
                construct_2D_interpolation_kernel(
                    ("Z", "Y"),
                    "temperature",
                    "temp_field",
                    N=invalid_n,  # type: ignore[arg-type]
                )

    def test_rejects_wrong_length_D_at_construction(self) -> None:
        for invalid_d in [(0,), (0, 0, 0)]:
            with pytest.raises(ValueError, match="D must be a tuple of two integers"):
                construct_2D_interpolation_kernel(
                    ("Z", "Y"),
                    "temperature",
                    "temp_field",
                    D=invalid_d,  # type: ignore[arg-type]
                )

    def test_mixed_derivative_order_differentiates_only_the_requested_axis(self, run_bound_kernel) -> None:
        N0, N1 = 2, 2
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", N=(N0, N1), D=(1, 0))
        degree0, degree1 = 2 * N0 - 1, 2 * N1 - 1
        p0 = np.polynomial.Polynomial(np.arange(1, degree0 + 2, dtype=np.float64))
        p1 = np.polynomial.Polynomial(np.arange(1, degree1 + 2, dtype=np.float64))
        dp0 = p0.deriv(1)

        idx0, idx1 = 10.25, 8.625
        I0, local0 = _stencil_origin_and_local_position(idx0, N0)
        I1, local1 = _stencil_origin_and_local_position(idx1, N1)
        n0, n1 = 2 * N0, 2 * N1
        grid0 = np.arange(n0 + 20, dtype=np.float64) - I0
        grid1 = np.arange(n1 + 20, dtype=np.float64) - I1
        mesh0, mesh1 = np.meshgrid(grid0, grid1, indexing="ij")
        field = p0(mesh0) * p1(mesh1)

        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([idx0], dtype=np.float64),
            "yidx": np.array([idx1], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (0.0, 0.0))}
        run_bound_kernel(kernel, particle_properties, {}, field_data)

        expected = dp0(local0) * p1(local1)
        assert particle_properties["temperature"][0] == pytest.approx(expected, rel=1e-8, abs=1e-8)

    def test_matches_particle_factory_directly(self, run_bound_kernel) -> None:
        kernel = construct_2D_interpolation_kernel(("Z", "Y"), "temperature", "temp_field", N=(2, 2))
        field = np.arange(40 * 48, dtype=np.float64).reshape(40, 48)
        offset0, offset1 = 0.1, -0.2
        idx0_value, idx1_value = 13.375, 15.625

        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([idx0_value], dtype=np.float64),
            "yidx": np.array([idx1_value], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (offset0, offset1))}
        run_bound_kernel(kernel, particle_properties, {}, field_data)

        single_impl = lagrange_2D_particle_factory(2, 2)
        expected = single_impl(
            field,
            np.float64(idx0_value + offset0),
            np.float64(idx1_value + offset1),
            field.shape[0] - 4,
            field.shape[1] - 4,
        )

        assert particle_properties["temperature"][0] == pytest.approx(expected)


class TestConstructTrilinearInterpolationKernel:
    """Tests for construct_3D_interpolation_kernel."""

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
        assert kernel.kernel.field_data["field"].dtype_constraints == (np.float64,)

    def test_custom_field_dtype_is_applied(self) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", field_dtype=np.float32)
        assert kernel.kernel.field_data["field"].dtype_constraints == (np.float32,)

    def test_output_dtype_defaults_to_field_dtype(self) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", field_dtype=np.float32)
        assert kernel.kernel.particle_properties["output"].dtype_constraints == (np.float32,)

    def test_custom_output_dtype_is_applied(self) -> None:
        kernel = construct_3D_interpolation_kernel(
            ("Z", "Y", "X"), "temperature", "temp_field", field_dtype=np.float64, output_dtype=np.float32
        )
        assert kernel.kernel.particle_properties["output"].dtype_constraints == (np.float32,)

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

    def test_passes_N_D_and_accumulate_to_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {}

        def fake_factory(N0: int, N1: int, N2: int, D0: int = 0, D1: int = 0, D2: int = 0, accumulate: bool = False):
            called["N0"] = N0
            called["N1"] = N1
            called["N2"] = N2
            called["D0"] = D0
            called["D1"] = D1
            called["D2"] = D2
            called["accumulate"] = accumulate

            def kernel_function_impl(*args):
                return None

            return kernel_function_impl

        monkeypatch.setattr(_kernel_constructors, "_lagrange_3D_kernel_function_factory", fake_factory)

        construct_3D_interpolation_kernel(
            ("Z", "Y", "X"), "temperature", "temp_field", N=(2, 3, 4), D=(1, 0, 0), accumulate=True
        )
        assert called == {"N0": 2, "N1": 3, "N2": 4, "D0": 1, "D1": 0, "D2": 0, "accumulate": True}

    def test_executes_and_interpolates_correctly(self, run_bound_kernel) -> None:
        """End-to-end: actually run the constructed kernel against an affine field.

        Exercises the `@kernel_function` decorator's key-matching for the "idx_0"/"idx_1"/"idx_2"
        declared particle properties, which a monkeypatched-factory test cannot catch.
        """
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", N=(1, 1, 1))
        shape = (4, 4, 4)
        field = np.fromfunction(lambda i, j, k: 7.0 + 1.0 * i + 10.0 * j + 100.0 * k, shape, dtype=np.float64)
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([1.25], dtype=np.float64),
            "yidx": np.array([2.5], dtype=np.float64),
            "xidx": np.array([0.75], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (0.0, 0.0, 0.0))}

        run_bound_kernel(kernel, particle_properties, {}, field_data)

        expected = 7.0 + 1.0 * 1.25 + 10.0 * 2.5 + 100.0 * 0.75
        assert particle_properties["temperature"][0] == pytest.approx(expected)

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_exact_for_degree_2n_minus_1_polynomial(self, run_bound_kernel, N: int) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", N=(N, N, N))
        degree = 2 * N - 1
        idx0, idx1, idx2 = 10.2, 8.4, 7.75
        I0, local0 = _stencil_origin_and_local_position(idx0, N)
        I1, local1 = _stencil_origin_and_local_position(idx1, N)
        I2, local2 = _stencil_origin_and_local_position(idx2, N)

        def polynomial(x0, x1, x2):
            total = 0.0
            for a in range(degree + 1):
                for b in range(degree + 1):
                    for c in range(degree + 1):
                        if a + b + c <= degree:
                            total += (a + 1) * (b + 2) * (c + 3) * (x0**a) * (x1**b) * (x2**c)
            return total

        grid0 = np.arange(28, dtype=np.float64) - I0
        grid1 = np.arange(24, dtype=np.float64) - I1
        grid2 = np.arange(20, dtype=np.float64) - I2
        mesh0, mesh1, mesh2 = np.meshgrid(grid0, grid1, grid2, indexing="ij")
        field = polynomial(mesh0, mesh1, mesh2)

        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([idx0], dtype=np.float64),
            "yidx": np.array([idx1], dtype=np.float64),
            "xidx": np.array([idx2], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (0.0, 0.0, 0.0))}
        run_bound_kernel(kernel, particle_properties, {}, field_data)

        expected = polynomial(local0, local1, local2)
        assert particle_properties["temperature"][0] == pytest.approx(expected, rel=1e-10, abs=1e-10)

    @pytest.mark.parametrize("shape", [(3, 4, 4), (4, 3, 4), (4, 4, 3), (3, 3, 4)])
    def test_rejects_field_smaller_than_stencil(self, run_bound_kernel, shape: tuple[int, int, int]) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", N=(2, 2, 2))
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([0.5], dtype=np.float64),
            "yidx": np.array([0.5], dtype=np.float64),
            "xidx": np.array([0.5], dtype=np.float64),
            "temperature": np.zeros(1, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(np.ones(shape, dtype=np.float64), (0.0, 0.0, 0.0))}

        with pytest.raises(ValueError, match=r"\(2 N0\) x \(2 N1\) x \(2 N2\) points"):
            run_bound_kernel(kernel, particle_properties, {}, field_data)

    def test_accepts_field_matching_stencil_size(self, run_bound_kernel) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", N=(2, 2, 2))
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([0.5], dtype=np.float64),
            "yidx": np.array([0.5], dtype=np.float64),
            "xidx": np.array([0.5], dtype=np.float64),
            "temperature": np.zeros(1, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(np.ones((4, 4, 4), dtype=np.float64), (0.0, 0.0, 0.0))}

        run_bound_kernel(kernel, particle_properties, {}, field_data)

    def test_accepts_field_with_different_size_per_axis(self, run_bound_kernel) -> None:
        """N is now per-axis, so the minimum field size is no longer symmetric."""
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", N=(1, 2, 3))
        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([0.5], dtype=np.float64),
            "yidx": np.array([1.5], dtype=np.float64),
            "xidx": np.array([2.5], dtype=np.float64),
            "temperature": np.zeros(1, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(np.ones((2, 4, 6), dtype=np.float64), (0.0, 0.0, 0.0))}

        run_bound_kernel(kernel, particle_properties, {}, field_data)

    @pytest.mark.parametrize(("N0", "N1"), [(1, 3), (3, 1)])
    def test_axis_0_exactness_depends_only_on_N0(self, run_bound_kernel, N0: int, N1: int) -> None:
        """A field that only varies along axis 0 isolates axis 0's own stencil capacity."""
        N2 = 2
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", N=(N0, N1, N2))
        degree = 2 * N0 - 1
        polynomial = np.polynomial.Polynomial(np.arange(1, degree + 2, dtype=np.float64))
        idx0 = 14.25
        I0, local0 = _stencil_origin_and_local_position(idx0, N0)
        n0, n1, n2 = 2 * N0, 2 * N1, 2 * N2
        field = np.tile(polynomial(np.arange(n0 + 20, dtype=np.float64) - I0)[:, None, None], (1, n1 + 20, n2 + 20))

        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([idx0], dtype=np.float64),
            "yidx": np.array([16.625], dtype=np.float64),
            "xidx": np.array([8.4], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (0.0, 0.0, 0.0))}
        run_bound_kernel(kernel, particle_properties, {}, field_data)

        assert particle_properties["temperature"][0] == pytest.approx(polynomial(local0), rel=1e-9, abs=1e-9)

    def test_rejects_nonpositive_N_on_any_axis_at_construction(self) -> None:
        for invalid_n in (0, -1):
            for N in ((invalid_n, 2, 2), (2, invalid_n, 2), (2, 2, invalid_n)):
                with pytest.raises(ValueError, match="N must be a positive integer"):
                    construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", N=N)

    def test_rejects_negative_D_on_any_axis_at_construction(self) -> None:
        for D in ((-1, 0, 0), (0, -1, 0), (0, 0, -1)):
            with pytest.raises(ValueError, match="derivative_order"):
                construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", D=D)

    def test_rejects_wrong_length_N_at_construction(self) -> None:
        for invalid_n in [(1, 1), (1, 1, 1, 1)]:
            with pytest.raises(ValueError, match="N must be a tuple of three integers"):
                construct_3D_interpolation_kernel(
                    ("Z", "Y", "X"),
                    "temperature",
                    "temp_field",
                    N=invalid_n,  # type: ignore[arg-type]
                )

    def test_rejects_wrong_length_D_at_construction(self) -> None:
        for invalid_d in [(0, 0), (0, 0, 0, 0)]:
            with pytest.raises(ValueError, match="D must be a tuple of three integers"):
                construct_3D_interpolation_kernel(
                    ("Z", "Y", "X"),
                    "temperature",
                    "temp_field",
                    D=invalid_d,  # type: ignore[arg-type]
                )

    def test_mixed_derivative_order_differentiates_only_the_requested_axis(self, run_bound_kernel) -> None:
        N0, N1, N2 = 2, 2, 2
        kernel = construct_3D_interpolation_kernel(
            ("Z", "Y", "X"), "temperature", "temp_field", N=(N0, N1, N2), D=(1, 0, 0)
        )
        degree0, degree1, degree2 = 2 * N0 - 1, 2 * N1 - 1, 2 * N2 - 1
        p0 = np.polynomial.Polynomial(np.arange(1, degree0 + 2, dtype=np.float64))
        p1 = np.polynomial.Polynomial(np.arange(1, degree1 + 2, dtype=np.float64))
        p2 = np.polynomial.Polynomial(np.arange(1, degree2 + 2, dtype=np.float64))
        dp0 = p0.deriv(1)

        idx0, idx1, idx2 = 10.25, 8.625, 7.25
        I0, local0 = _stencil_origin_and_local_position(idx0, N0)
        I1, local1 = _stencil_origin_and_local_position(idx1, N1)
        I2, local2 = _stencil_origin_and_local_position(idx2, N2)
        n0, n1, n2 = 2 * N0, 2 * N1, 2 * N2
        grid0 = np.arange(n0 + 20, dtype=np.float64) - I0
        grid1 = np.arange(n1 + 20, dtype=np.float64) - I1
        grid2 = np.arange(n2 + 20, dtype=np.float64) - I2
        mesh0, mesh1, mesh2 = np.meshgrid(grid0, grid1, grid2, indexing="ij")
        field = p0(mesh0) * p1(mesh1) * p2(mesh2)

        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([idx0], dtype=np.float64),
            "yidx": np.array([idx1], dtype=np.float64),
            "xidx": np.array([idx2], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (0.0, 0.0, 0.0))}
        run_bound_kernel(kernel, particle_properties, {}, field_data)

        expected = dp0(local0) * p1(local1) * p2(local2)
        assert particle_properties["temperature"][0] == pytest.approx(expected, rel=1e-8, abs=1e-8)

    def test_matches_particle_factory_directly(self, run_bound_kernel) -> None:
        kernel = construct_3D_interpolation_kernel(("Z", "Y", "X"), "temperature", "temp_field", N=(2, 2, 2))
        field = np.arange(20 * 18 * 16, dtype=np.float64).reshape(20, 18, 16)
        offset0, offset1, offset2 = 0.15, -0.25, 0.05
        idx0_value, idx1_value, idx2_value = 9.125, 8.625, 7.25

        particle_properties = {
            "status": np.zeros(1, dtype=np.uint8),
            "zidx": np.array([idx0_value], dtype=np.float64),
            "yidx": np.array([idx1_value], dtype=np.float64),
            "xidx": np.array([idx2_value], dtype=np.float64),
            "temperature": np.full(1, np.nan, dtype=np.float64),
        }
        field_data = {"temp_field": FieldData(field, (offset0, offset1, offset2))}
        run_bound_kernel(kernel, particle_properties, {}, field_data)

        single_impl = lagrange_3D_particle_factory(2, 2, 2)
        expected = single_impl(
            field,
            np.float64(idx0_value + offset0),
            np.float64(idx1_value + offset1),
            np.float64(idx2_value + offset2),
            field.shape[0] - 4,
            field.shape[1] - 4,
            field.shape[2] - 4,
        )

        assert particle_properties["temperature"][0] == pytest.approx(expected)


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
