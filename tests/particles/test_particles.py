"""Tests for the particles module (_FrozenArrayMapping, Particles, ParticlesView)."""

import numpy as np
import pytest

from offline_particles.particles import Particles, ParticlesView, _FrozenArrayMapping

# ---------------------------------------------------------------------------
# _FrozenArrayMapping (tested indirectly through Particles)
# ---------------------------------------------------------------------------


class TestFrozenArrayMappingShapeCheck:
    def test_accepts_arrays_with_same_shape(self) -> None:
        p = Particles(5, {"x": np.dtype(np.float64), "y": np.dtype(np.float64)})
        assert p.shape == (5,)

    def test_rejects_arrays_with_different_shapes(self) -> None:
        # Particles always creates arrays of the same shape, so test _FrozenArrayMapping
        # via direct construction with inconsistent arrays through the inherited path.
        # Easiest: subclass or use the fact that Particles itself enforces uniform shape.
        # The shape mismatch path is exercised when constructing via **arrays that differ.
        a = np.zeros((3,))
        b = np.zeros((4,))
        with pytest.raises(ValueError, match="same shape"):
            _FrozenArrayMapping({"a": a, "b": b})

    def test_shape_property(self) -> None:
        p = Particles(7, {"x": np.dtype(np.float64)})
        assert p.shape == (7,)

    def test_dtypes_property(self) -> None:
        p = Particles(3, {"x": np.dtype(np.float64), "y": np.dtype(np.int32)})
        assert p.dtypes["x"] == np.dtype(np.float64)
        assert p.dtypes["y"] == np.dtype(np.int32)

    def test_arrays_property(self) -> None:
        p = Particles(4, {"val": np.dtype(np.float32)})
        assert "val" in p.arrays
        assert p.arrays["val"].dtype == np.dtype(np.float32)

    def test_getattr_returns_array(self) -> None:
        p = Particles(5, {"x": np.dtype(np.float64)})
        assert isinstance(p.x, np.ndarray)

    def test_getattr_missing_raises(self) -> None:
        p = Particles(5, {"x": np.dtype(np.float64)})
        with pytest.raises(AttributeError):
            _ = p.nonexistent

    def test_getitem_returns_array(self) -> None:
        p = Particles(5, {"x": np.dtype(np.float64)})
        assert isinstance(p["x"], np.ndarray)

    def test_getitem_missing_raises(self) -> None:
        p = Particles(5, {"x": np.dtype(np.float64)})
        with pytest.raises(KeyError):
            _ = p["nonexistent"]

    def test_setattr_raises(self) -> None:
        p = Particles(5, {"x": np.dtype(np.float64)})
        with pytest.raises(AttributeError):
            p.x = np.zeros(5)  # type: ignore[misc]

    def test_repr(self) -> None:
        p = Particles(3, {"x": np.dtype(np.float64)})
        r = repr(p)
        assert "shape" in r
        assert "float64" in r

    def test_str_with_no_hidden_fields(self) -> None:
        p = Particles(3, {"x": np.dtype(np.float64)})
        s = str(p)
        assert "x" in s

    def test_str_with_hidden_fields(self) -> None:
        p = Particles(3, {"x": np.dtype(np.float64), "_hidden": np.dtype(np.float64)})
        s = str(p)
        assert "hidden" in s


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------


class TestParticles:
    def test_creates_zero_arrays(self) -> None:
        p = Particles(4, {"x": np.dtype(np.float64)})
        np.testing.assert_array_equal(p["x"], np.zeros(4))

    def test_len(self) -> None:
        p = Particles(10, {"x": np.dtype(np.float64)})
        assert len(p) == 10

    def test_arrays_are_writable(self) -> None:
        p = Particles(5, {"x": np.dtype(np.float64)})
        p["x"][0] = 42.0
        assert p["x"][0] == 42.0

    def test_multiple_properties(self) -> None:
        p = Particles(3, {"x": np.dtype(np.float64), "y": np.dtype(np.float32), "status": np.dtype(np.uint8)})
        assert "x" in p.arrays
        assert "y" in p.arrays
        assert "status" in p.arrays
        assert p["x"].dtype == np.float64
        assert p["y"].dtype == np.float32
        assert p["status"].dtype == np.uint8

    def test_zero_particles(self) -> None:
        p = Particles(0, {"x": np.dtype(np.float64)})
        assert len(p) == 0
        assert p.shape == (0,)


# ---------------------------------------------------------------------------
# ParticlesView
# ---------------------------------------------------------------------------


class TestParticlesView:
    def test_view_matches_parent_values(self) -> None:
        p = Particles(5, {"x": np.dtype(np.float64)})
        p["x"][:] = np.arange(5, dtype=np.float64)
        view = ParticlesView(p)
        np.testing.assert_array_equal(view["x"], p["x"])

    def test_view_reflects_parent_changes(self) -> None:
        p = Particles(5, {"x": np.dtype(np.float64)})
        view = ParticlesView(p)
        p["x"][2] = 99.0
        assert view["x"][2] == 99.0

    def test_view_is_read_only(self) -> None:
        p = Particles(5, {"x": np.dtype(np.float64)})
        view = ParticlesView(p)
        with pytest.raises(ValueError, match="read-only"):
            view["x"][0] = 1.0

    def test_view_len(self) -> None:
        p = Particles(8, {"x": np.dtype(np.float64)})
        view = ParticlesView(p)
        assert len(view) == 8

    def test_view_shape(self) -> None:
        p = Particles(6, {"x": np.dtype(np.float64)})
        view = ParticlesView(p)
        assert view.shape == (6,)

    def test_view_dtypes(self) -> None:
        p = Particles(3, {"x": np.dtype(np.float64)})
        view = ParticlesView(p)
        assert view.dtypes["x"] == np.dtype(np.float64)

    def test_view_getattr(self) -> None:
        p = Particles(4, {"x": np.dtype(np.float64)})
        view = ParticlesView(p)
        assert isinstance(view.x, np.ndarray)
