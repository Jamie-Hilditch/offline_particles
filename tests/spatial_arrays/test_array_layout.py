"""Tests for the ArrayLayout class."""

import pytest

from offline_particles.spatial_arrays import ArrayAxis, ArrayLayout, Stagger


class TestArrayLayoutConstruction:
    def test_3d_zyx_construction(self) -> None:
        layout = ArrayLayout(("Z", "Y", "X"), ("center", "center", "center"))
        assert layout.ndim == 3
        assert layout.axes == (ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X)
        assert layout.staggers == (Stagger.CENTER, Stagger.CENTER, Stagger.CENTER)

    def test_2d_yx_construction(self) -> None:
        layout = ArrayLayout(("Y", "X"), ("left", "right"))
        assert layout.ndim == 2
        assert layout.axes == (ArrayAxis.Y, ArrayAxis.X)
        assert layout.staggers == (Stagger.LEFT, Stagger.RIGHT)

    def test_1d_z_construction(self) -> None:
        layout = ArrayLayout(("Z",), ("outer",))
        assert layout.ndim == 1
        assert layout.axes == (ArrayAxis.Z,)
        assert layout.staggers == (Stagger.OUTER,)

    def test_accepts_enum_members_directly(self) -> None:
        layout = ArrayLayout((ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X), (Stagger.CENTER, Stagger.LEFT, Stagger.RIGHT))
        assert layout.axes == (ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X)
        assert layout.staggers == (Stagger.CENTER, Stagger.LEFT, Stagger.RIGHT)

    def test_aliases_resolve_to_canonical_axes(self) -> None:
        layout = ArrayLayout(("DEPTH", "LATITUDE", "LON"), ("center", "center", "center"))
        assert layout.axes == (ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X)

    def test_arbitrary_axis_ordering(self) -> None:
        """Axes can be in any order, not just Z-Y-X."""
        layout = ArrayLayout(("X", "Y", "Z"), ("center", "center", "center"))
        assert layout.axes == (ArrayAxis.X, ArrayAxis.Y, ArrayAxis.Z)

    def test_non_zyx_ordering_2d(self) -> None:
        """2D layout does not have to be Y-X; Z-X is also valid."""
        layout = ArrayLayout(("Z", "X"), ("center", "center"))
        assert layout.axes == (ArrayAxis.Z, ArrayAxis.X)


class TestArrayLayoutNdim:
    def test_ndim_matches_number_of_axes(self) -> None:
        for n_axes in range(1, 4):
            axes = ("Z", "Y", "X")[:n_axes]
            staggers = ("center",) * n_axes
            layout = ArrayLayout(axes, staggers)
            assert layout.ndim == n_axes


class TestArrayLayoutOffsets:
    def test_offsets_match_stagger_offsets(self) -> None:
        layout = ArrayLayout(("Z", "Y", "X"), ("center", "left", "right"))
        assert layout.offsets == (Stagger.CENTER.offset, Stagger.LEFT.offset, Stagger.RIGHT.offset)

    def test_center_stagger_gives_zero_offset(self) -> None:
        layout = ArrayLayout(("Z",), ("center",))
        assert layout.offsets == (0.0,)

    def test_all_stagger_types_produce_correct_offsets(self) -> None:
        for stagger in Stagger:
            layout = ArrayLayout(("Z",), (stagger,))
            assert layout.offsets == (stagger.offset,)


class TestArrayLayoutImmutability:
    def test_cannot_set_attribute(self) -> None:
        layout = ArrayLayout(("Z", "Y", "X"), ("center", "center", "center"))
        with pytest.raises(AttributeError):
            layout.ndim = 2  # type: ignore[misc]

    def test_cannot_set_axes(self) -> None:
        layout = ArrayLayout(("Z", "Y", "X"), ("center", "center", "center"))
        with pytest.raises(AttributeError):
            layout.axes = (ArrayAxis.Y, ArrayAxis.X)  # type: ignore[misc]

    def test_cannot_delete_attribute(self) -> None:
        layout = ArrayLayout(("Z", "Y", "X"), ("center", "center", "center"))
        with pytest.raises(AttributeError):
            del layout.ndim  # type: ignore[misc]


class TestArrayLayoutValidation:
    def test_mismatched_axes_and_staggers_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="Number of axes and staggers must match"):
            ArrayLayout(("Z", "Y"), ("center",))

    def test_duplicate_axes_raise(self) -> None:
        with pytest.raises(ValueError, match="Axes must be unique"):
            ArrayLayout(("Z", "Z"), ("center", "center"))

    def test_duplicate_axes_raise_3d(self) -> None:
        with pytest.raises(ValueError, match="Axes must be unique"):
            ArrayLayout(("Z", "Y", "Z"), ("center", "center", "center"))

    def test_alias_duplicate_axes_raise(self) -> None:
        """Mapping two dims to the same canonical axis should raise."""
        with pytest.raises(ValueError, match="Axes must be unique"):
            ArrayLayout(("Z", "DEPTH"), ("center", "center"))

    def test_invalid_axis_string_raises(self) -> None:
        with pytest.raises(ValueError):
            ArrayLayout(("INVALID",), ("center",))

    def test_invalid_stagger_string_raises(self) -> None:
        with pytest.raises(ValueError):
            ArrayLayout(("Z",), ("invalid_stagger",))
