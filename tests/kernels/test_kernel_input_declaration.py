import numpy as np
import pytest

from offline_particles.kernels._kernels import KernelInputDeclaration
from offline_particles.kernels.input_declarations import (
    STATUS_DECLARATION,
    XIDX_DECLARATION,
    YIDX_DECLARATION,
    ZIDX_DECLARATION,
)
from offline_particles.particles import _REQUIRED_PARTICLE_PROPERTY_FIELDS


class TestKernelInputDeclarationConstruction:
    def test_single_dtype_constraint_is_stored_as_tuple(self) -> None:
        declaration = KernelInputDeclaration("x", np.float32)

        assert declaration.dtype_constraints == (np.float32,)

    def test_iterable_dtype_constraints_are_stored_as_tuple(self) -> None:
        declaration = KernelInputDeclaration("x", [np.float32, np.float64])

        assert declaration.dtype_constraints == (np.float32, np.float64)

    def test_invalid_dtype_constraint_type_raises(self) -> None:
        with pytest.raises(TypeError, match="subtype of np.generic"):
            KernelInputDeclaration("x", int)  # type: ignore[invalid-argument-type]

    def test_invalid_dtype_constraint_object_raises(self) -> None:
        with pytest.raises(TypeError, match="subtype of np.generic"):
            KernelInputDeclaration("x", (np.float32, np.dtype(np.float64)))  # type: ignore[invalid-argument-type]


class TestKernelInputDeclarationValidateDtype:
    def test_accepts_exact_dtype_type(self) -> None:
        declaration = KernelInputDeclaration("x", np.float32)

        declaration.validate_dtype(np.float32)

    def test_accepts_numpy_dtype_object(self) -> None:
        declaration = KernelInputDeclaration("x", np.float32)

        declaration.validate_dtype(np.dtype(np.float32))

    def test_accepts_abstract_dtype_constraint(self) -> None:
        declaration = KernelInputDeclaration("x", np.floating)

        declaration.validate_dtype(np.float32)

    def test_accepts_multiple_dtype_constraints(self) -> None:
        declaration = KernelInputDeclaration("x", (np.float32, np.int32))

        declaration.validate_dtype(np.int32)

    def test_accepts_exact_and_abstract_dtype_constraints(self) -> None:
        declaration = KernelInputDeclaration("x", (np.float32, np.integer))

        declaration.validate_dtype(np.float32)
        declaration.validate_dtype(np.int32)

    def test_rejects_incompatible_dtype(self) -> None:
        declaration = KernelInputDeclaration("x", np.float32)

        with pytest.raises(TypeError, match="dtype constraints"):
            declaration.validate_dtype(np.int32)


class TestKernelInputDeclarationSummaryAndDescription:
    def test_summary_starts_with_name(self) -> None:
        declaration = KernelInputDeclaration("complicated_name", (np.float32, np.int32))

        assert declaration.summary.startswith("complicated_name")

    def test_summary_includes_dtype_constraints(self) -> None:
        declaration = KernelInputDeclaration("x", (np.float32, np.int32))

        assert "float32" in declaration.summary
        assert "int32" in declaration.summary

    def test_description_includes_additional_description_if_provided(self) -> None:
        declaration = KernelInputDeclaration("x", np.float32, description="This is x.")

        assert "This is x." in declaration.description

    def test_description_returns_summary_if_no_description_provided(self) -> None:
        declaration = KernelInputDeclaration("x", np.float32)

        assert declaration.description == declaration.summary

    def test_compare_equal_ignores_description(self) -> None:
        declaration1 = KernelInputDeclaration("x", np.float32, description="First description.")
        declaration2 = KernelInputDeclaration("x", np.float32, description="Second description.")

        assert declaration1 == declaration2


class TestRequiredParticlePropertyDeclarations:
    @pytest.mark.parametrize(
        ("declaration", "expected_name", "expected_dtype"),
        [
            (STATUS_DECLARATION, "status", np.dtype(np.uint8)),
            (ZIDX_DECLARATION, "zidx", np.dtype(np.float64)),
            (YIDX_DECLARATION, "yidx", np.dtype(np.float64)),
            (XIDX_DECLARATION, "xidx", np.dtype(np.float64)),
        ],
    )
    def test_required_particle_property_declarations_match_particles_module(
        self,
        declaration: KernelInputDeclaration,
        expected_name: str,
        expected_dtype: np.dtype,
    ) -> None:
        assert declaration.name == expected_name
        assert declaration.dtype_constraints == (expected_dtype.type,)
        assert _REQUIRED_PARTICLE_PROPERTY_FIELDS[expected_name] == expected_dtype

    def test_required_particle_property_names_match_particles_module(self) -> None:
        assert {
            STATUS_DECLARATION.name,
            ZIDX_DECLARATION.name,
            YIDX_DECLARATION.name,
            XIDX_DECLARATION.name,
        } == set(_REQUIRED_PARTICLE_PROPERTY_FIELDS)
