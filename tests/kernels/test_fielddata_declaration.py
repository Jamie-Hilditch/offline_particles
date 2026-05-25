import numpy as np
import numpy.typing as npt
import pytest

from offline_particles.fields import StaticField
from offline_particles.kernels import FieldDataDeclaration


def _make_field(dtype: npt.DTypeLike) -> StaticField:
    arr = np.zeros((3,), dtype=dtype)
    # 1D field with Z axis and centered stagger
    return StaticField.from_arraylike(arr, axes=("Z",), staggers=("center",))


def test_validate_field_dtype_mismatch_raises() -> None:
    decl = FieldDataDeclaration("field", np.float32)
    field = _make_field(np.int32)
    with pytest.raises(TypeError):
        decl.validate_field(field)


def test_validate_field_accepts_exact_match_dtype() -> None:
    decl = FieldDataDeclaration("field", np.float32)
    field = _make_field(np.float32)
    # should not raise
    decl.validate_field(field)


def test_validate_field_accepts_compatible_dtypes() -> None:
    # Test that the validator accepts dtypes that are compatible with the declared dtype
    # For example, float64 should be accepted when float32 is declared (since float64 is a superset of float32)
    decl = FieldDataDeclaration("field", np.inexact)
    field = _make_field(np.float64)
    # should not raise
    decl.validate_field(field)


def test_validate_field_rejects_incompatible_dtypes() -> None:
    # Test that the validator rejects dtypes that are not compatible with the declared dtype
    # For example, int32 should not be accepted when inexact is declared
    decl = FieldDataDeclaration("field", np.inexact)
    field = _make_field(np.int32)
    with pytest.raises(TypeError):
        decl.validate_field(field)


def test_validate_field_accepts_multiple_compatible_dtypes() -> None:
    # Test that the validator accepts multiple dtypes that are compatible with the declared dtype
    # For example, both float32 and float64 should be accepted when inexact is declared
    decl = FieldDataDeclaration("field", (np.float32, np.float64))
    field1 = _make_field(np.float32)
    field2 = _make_field(np.float64)
    # should not raise for either field
    decl.validate_field(field1)
    decl.validate_field(field2)
