"""Tests for layout validator functions."""

import pytest

from offline_particles.kernels.layout_validators import (
    ordering_validator_factory,
    validate_X_ordering,
    validate_Y_ordering,
    validate_YX_ordering,
    validate_Z_ordering,
    validate_ZX_ordering,
    validate_ZY_ordering,
    validate_ZYX_ordering,
)
from offline_particles.spatial_arrays import ArrayLayout


def _layout(*axes: str, stagger: str = "center") -> ArrayLayout:
    """Helper: build a layout from axis names all with the same stagger."""
    return ArrayLayout(axes, (stagger,) * len(axes))


class TestValidateZYXOrdering:
    def test_passes_for_zyx(self) -> None:
        validate_ZYX_ordering(_layout("Z", "Y", "X"))

    def test_fails_for_xyz(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_ZYX_ordering(_layout("X", "Y", "Z"))

    def test_fails_for_yx(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_ZYX_ordering(_layout("Y", "X"))

    def test_fails_for_z_only(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_ZYX_ordering(_layout("Z"))

    def test_passes_regardless_of_staggers(self) -> None:
        for stagger in ("center", "left", "right", "inner", "outer"):
            validate_ZYX_ordering(_layout("Z", "Y", "X", stagger=stagger))


class TestValidateZYOrdering:
    def test_passes_for_zy(self) -> None:
        validate_ZY_ordering(_layout("Z", "Y"))

    def test_fails_for_yz(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_ZY_ordering(_layout("Y", "Z"))

    def test_fails_for_zyx(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_ZY_ordering(_layout("Z", "Y", "X"))

    def test_fails_for_z_only(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_ZY_ordering(_layout("Z"))

    def test_passes_regardless_of_staggers(self) -> None:
        for stagger in ("center", "left", "right", "inner", "outer"):
            validate_ZY_ordering(_layout("Z", "Y", stagger=stagger))


class TestValidateYXOrdering:
    def test_passes_for_yx(self) -> None:
        validate_YX_ordering(_layout("Y", "X"))

    def test_fails_for_xy(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_YX_ordering(_layout("X", "Y"))

    def test_fails_for_zyx(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_YX_ordering(_layout("Z", "Y", "X"))

    def test_fails_for_y_only(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_YX_ordering(_layout("Y"))

    def test_passes_regardless_of_staggers(self) -> None:
        for stagger in ("center", "left", "right", "inner", "outer"):
            validate_YX_ordering(_layout("Y", "X", stagger=stagger))


class TestValidateZXOrdering:
    def test_passes_for_zx(self) -> None:
        validate_ZX_ordering(_layout("Z", "X"))

    def test_fails_for_xz(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_ZX_ordering(_layout("X", "Z"))

    def test_fails_for_zyx(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_ZX_ordering(_layout("Z", "Y", "X"))

    def test_fails_for_z_only(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_ZX_ordering(_layout("Z"))

    def test_passes_regardless_of_staggers(self) -> None:
        for stagger in ("center", "left", "right", "inner", "outer"):
            validate_ZX_ordering(_layout("Z", "X", stagger=stagger))


class TestValidateZOrdering:
    def test_passes_for_z(self) -> None:
        validate_Z_ordering(_layout("Z"))

    def test_fails_for_y(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_Z_ordering(_layout("Y"))

    def test_fails_for_zy(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_Z_ordering(_layout("Z", "Y"))

    def test_passes_regardless_of_stagger(self) -> None:
        for stagger in ("center", "left", "right", "inner", "outer"):
            validate_Z_ordering(_layout("Z", stagger=stagger))


class TestValidateYOrdering:
    def test_passes_for_y(self) -> None:
        validate_Y_ordering(_layout("Y"))

    def test_fails_for_z(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_Y_ordering(_layout("Z"))

    def test_fails_for_yx(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_Y_ordering(_layout("Y", "X"))

    def test_passes_regardless_of_stagger(self) -> None:
        for stagger in ("center", "left", "right", "inner", "outer"):
            validate_Y_ordering(_layout("Y", stagger=stagger))


class TestValidateXOrdering:
    def test_passes_for_x(self) -> None:
        validate_X_ordering(_layout("X"))

    def test_fails_for_y(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_X_ordering(_layout("Y"))

    def test_fails_for_yx(self) -> None:
        with pytest.raises(ValueError, match="Expected axes"):
            validate_X_ordering(_layout("Y", "X"))

    def test_passes_regardless_of_stagger(self) -> None:
        for stagger in ("center", "left", "right", "inner", "outer"):
            validate_X_ordering(_layout("X", stagger=stagger))


class TestLayoutValidatorsWithFieldDataDeclaration:
    """Integration tests: FieldDataDeclaration.validate_field uses layout_validators."""

    def test_validate_field_passes_with_correct_layout(self) -> None:
        import numpy as np

        from offline_particles.fields import StaticField
        from offline_particles.kernels import FieldDataDeclaration

        data = np.ones((4, 5, 6), dtype=np.float64)
        field = StaticField.from_numpy(data, ("Z", "Y", "X"), ("center", "center", "center"))
        decl = FieldDataDeclaration("u", np.float64, [validate_ZYX_ordering])
        decl.validate_field(field)  # should not raise

    def test_validate_field_raises_on_wrong_layout(self) -> None:
        import numpy as np

        from offline_particles.fields import StaticField
        from offline_particles.kernels import FieldDataDeclaration

        data = np.ones((5, 6), dtype=np.float64)
        field = StaticField.from_numpy(data, ("Y", "X"), ("center", "center"))
        decl = FieldDataDeclaration("u", np.float64, [validate_ZYX_ordering])
        with pytest.raises(ValueError, match="Expected axes"):
            decl.validate_field(field)

    def test_validate_field_passes_with_no_validators(self) -> None:
        import numpy as np

        from offline_particles.fields import StaticField
        from offline_particles.kernels import FieldDataDeclaration

        data = np.ones((5, 6), dtype=np.float64)
        field = StaticField.from_numpy(data, ("Y", "X"), ("center", "center"))
        decl = FieldDataDeclaration("u", np.float64)
        decl.validate_field(field)  # no validators → always passes

    def test_validate_field_raises_on_dtype_mismatch(self) -> None:
        import numpy as np

        from offline_particles.fields import StaticField
        from offline_particles.kernels import FieldDataDeclaration

        data = np.ones((4, 5, 6), dtype=np.float32)
        field = StaticField.from_numpy(data, ("Z", "Y", "X"), ("center", "center", "center"))
        decl = FieldDataDeclaration("u", np.float64, [validate_ZYX_ordering])
        with pytest.raises(TypeError, match="dtype"):
            decl.validate_field(field)

    def test_multiple_validators_all_checked(self) -> None:
        import numpy as np

        from offline_particles.fields import StaticField
        from offline_particles.kernels import FieldDataDeclaration

        # YX layout: passes validate_YX_ordering but fails validate_ZYX_ordering
        data = np.ones((5, 6), dtype=np.float64)
        field = StaticField.from_numpy(data, ("Y", "X"), ("center", "center"))
        decl = FieldDataDeclaration("u", np.float64, [validate_YX_ordering, validate_ZYX_ordering])
        with pytest.raises(ValueError, match="Expected axes"):
            decl.validate_field(field)


class TestOrderingValidatorFactory:
    def test_factory_returns_callable(self) -> None:
        from offline_particles.spatial_arrays import ArrayAxis

        validator = ordering_validator_factory((ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X))
        assert callable(validator)

    def test_factory_validator_passes_for_matching_axes(self) -> None:
        from offline_particles.spatial_arrays import ArrayAxis

        validator = ordering_validator_factory((ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X))
        validator(_layout("Z", "Y", "X"))  # should not raise

    def test_factory_validator_fails_for_wrong_order(self) -> None:
        from offline_particles.spatial_arrays import ArrayAxis

        validator = ordering_validator_factory((ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X))
        with pytest.raises(ValueError, match="Expected axes"):
            validator(_layout("X", "Y", "Z"))

    def test_factory_validator_fails_for_wrong_number_of_axes(self) -> None:
        from offline_particles.spatial_arrays import ArrayAxis

        validator = ordering_validator_factory((ArrayAxis.Z, ArrayAxis.Y))
        with pytest.raises(ValueError, match="Expected axes"):
            validator(_layout("Z", "Y", "X"))

    def test_factory_validator_passes_regardless_of_staggers(self) -> None:
        from offline_particles.spatial_arrays import ArrayAxis

        validator = ordering_validator_factory((ArrayAxis.Y, ArrayAxis.X))
        for stagger in ("center", "left", "right", "inner", "outer"):
            validator(_layout("Y", "X", stagger=stagger))  # should not raise

    def test_factory_validator_for_single_axis(self) -> None:
        from offline_particles.spatial_arrays import ArrayAxis

        validator = ordering_validator_factory((ArrayAxis.X,))
        validator(_layout("X"))  # should not raise
        with pytest.raises(ValueError, match="Expected axes"):
            validator(_layout("Y"))

    def test_factory_validator_non_standard_ordering(self) -> None:
        """Factory can create validators for orderings with no hand-written equivalent."""
        from offline_particles.spatial_arrays import ArrayAxis

        validator = ordering_validator_factory((ArrayAxis.X, ArrayAxis.Y, ArrayAxis.Z))
        validator(_layout("X", "Y", "Z"))  # should not raise
        with pytest.raises(ValueError, match="Expected axes"):
            validator(_layout("Z", "Y", "X"))

