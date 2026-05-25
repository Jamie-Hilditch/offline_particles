import numpy as np
import pytest

from offline_particles.kernels._kernels import KernelInputDeclaration


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
