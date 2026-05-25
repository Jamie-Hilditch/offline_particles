"""Tests for ParticleKernel and BoundKernel."""

import numpy as np
import pytest

from offline_particles.kernels import (
    BoundKernel,
    FieldDataDeclaration,
    ParticleKernel,
    ParticlePropertyDeclaration,
    ScalarDeclaration,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop(pp, sc, fd) -> None:
    pass


def _noop2(pp, sc, fd) -> None:
    pass


def _make_simple_kernel() -> ParticleKernel:
    return ParticleKernel(
        _noop,
        [ParticlePropertyDeclaration("x", np.float64), ParticlePropertyDeclaration("status", np.uint8)],
        [ScalarDeclaration("_dt", np.float64)],
    )


# ---------------------------------------------------------------------------
# ParticleKernel construction
# ---------------------------------------------------------------------------


class TestParticleKernelConstruction:
    def test_single_function(self) -> None:
        kernel = ParticleKernel(_noop)
        assert len(kernel.functions) == 1

    def test_list_of_functions(self) -> None:
        kernel = ParticleKernel([_noop, _noop2])
        assert len(kernel.functions) == 2

    def test_particle_properties_stored(self) -> None:
        decl = ParticlePropertyDeclaration("x", np.float64)
        kernel = ParticleKernel(_noop, [decl])
        assert "x" in kernel.particle_properties
        assert kernel.particle_properties["x"].dtype_constraints == (np.float64,)

    def test_scalars_stored(self) -> None:
        decl = ScalarDeclaration("_dt", np.float64)
        kernel = ParticleKernel(_noop, scalars=[decl])
        assert "_dt" in kernel.scalars

    def test_field_data_stored(self) -> None:
        decl = FieldDataDeclaration("u", np.float64)
        kernel = ParticleKernel(_noop, field_data=[decl])
        assert "u" in kernel.field_data

    def test_duplicate_particle_property_raises(self) -> None:
        decl = ParticlePropertyDeclaration("x", np.float64)
        with pytest.raises(ValueError, match="Duplicate"):
            ParticleKernel(_noop, [decl, decl])

    def test_duplicate_scalar_raises(self) -> None:
        decl = ScalarDeclaration("_dt", np.float64)
        with pytest.raises(ValueError, match="Duplicate"):
            ParticleKernel(_noop, scalars=[decl, decl])

    def test_non_callable_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            ParticleKernel([_noop, "not_callable"])  # type: ignore[list-item]

    def test_repr(self) -> None:
        kernel = _make_simple_kernel()
        r = repr(kernel)
        assert "ParticleKernel" in r

    def test_str(self) -> None:
        kernel = _make_simple_kernel()
        s = str(kernel)
        assert "Particle Kernel" in s

    def test_docstring_built(self) -> None:
        kernel = _make_simple_kernel()
        assert kernel.__doc__ is not None


# ---------------------------------------------------------------------------
# ParticleKernel calling
# ---------------------------------------------------------------------------


class TestParticleKernelCall:
    def test_calls_function(self) -> None:
        called = []

        def recorder(pp, sc, fd):
            called.append(True)

        kernel = ParticleKernel(recorder)
        kernel({}, {}, {})
        assert called == [True]

    def test_calls_all_chained_functions(self) -> None:
        order = []

        def fn1(pp, sc, fd):
            order.append(1)

        def fn2(pp, sc, fd):
            order.append(2)

        kernel = ParticleKernel([fn1, fn2])
        kernel({}, {}, {})
        assert order == [1, 2]


# ---------------------------------------------------------------------------
# ParticleKernel chaining
# ---------------------------------------------------------------------------


class TestParticleKernelChaining:
    def test_chain_class_method(self) -> None:
        k1 = ParticleKernel(_noop, [ParticlePropertyDeclaration("x", np.float64)])
        k2 = ParticleKernel(_noop2, [ParticlePropertyDeclaration("y", np.float64)])
        chained = ParticleKernel.chain(k1, k2)
        assert "x" in chained.particle_properties
        assert "y" in chained.particle_properties

    def test_chain_with_method(self) -> None:
        k1 = ParticleKernel(_noop, [ParticlePropertyDeclaration("x", np.float64)])
        k2 = ParticleKernel(_noop2, [ParticlePropertyDeclaration("y", np.float64)])
        chained = k1.chain_with(k2)
        assert "x" in chained.particle_properties
        assert "y" in chained.particle_properties

    def test_chain_conflicting_declarations_raises(self) -> None:
        k1 = ParticleKernel(_noop, [ParticlePropertyDeclaration("x", np.float64)])
        k2 = ParticleKernel(_noop2, [ParticlePropertyDeclaration("x", np.int32)])
        with pytest.raises(ValueError, match="Conflicting"):
            ParticleKernel.chain(k1, k2)


# ---------------------------------------------------------------------------
# BoundKernel construction
# ---------------------------------------------------------------------------


class TestBoundKernelConstruction:
    def test_default_bindings_use_declared_names(self) -> None:
        kernel = _make_simple_kernel()
        bound = BoundKernel(kernel)
        assert bound.particle_property_bindings["x"] == "x"
        assert bound.particle_property_bindings["status"] == "status"
        assert bound.scalar_bindings["_dt"] == "_dt"

    def test_custom_particle_property_binding(self) -> None:
        kernel = ParticleKernel(_noop, [ParticlePropertyDeclaration("x", np.float64)])
        bound = BoundKernel(kernel, particle_property_bindings={"x": "xidx"})
        assert bound.particle_property_bindings["x"] == "xidx"

    def test_custom_scalar_binding(self) -> None:
        kernel = ParticleKernel(_noop, scalars=[ScalarDeclaration("_dt", np.float64)])
        bound = BoundKernel(kernel, scalar_bindings={"_dt": "my_dt"})
        assert bound.scalar_bindings["_dt"] == "my_dt"

    def test_unused_particle_property_binding_raises(self) -> None:
        kernel = ParticleKernel(_noop)
        with pytest.raises(ValueError, match="Unused"):
            BoundKernel(kernel, particle_property_bindings={"nonexistent": "y"})

    def test_unused_scalar_binding_raises(self) -> None:
        kernel = ParticleKernel(_noop)
        with pytest.raises(ValueError, match="Unused"):
            BoundKernel(kernel, scalar_bindings={"nonexistent": "my_dt"})

    def test_unused_field_data_binding_raises(self) -> None:
        kernel = ParticleKernel(_noop)
        with pytest.raises(ValueError, match="Unused"):
            BoundKernel(kernel, field_data_bindings={"nonexistent": "u"})

    def test_kernel_property(self) -> None:
        kernel = _make_simple_kernel()
        bound = BoundKernel(kernel)
        assert bound.kernel is kernel

    def test_docstring_built(self) -> None:
        kernel = _make_simple_kernel()
        bound = BoundKernel(kernel)
        assert bound.__doc__ is not None


# ---------------------------------------------------------------------------
# BoundKernel rebind
# ---------------------------------------------------------------------------


class TestBoundKernelRebind:
    def test_rebind_particle_property(self) -> None:
        kernel = ParticleKernel(_noop, [ParticlePropertyDeclaration("x", np.float64)])
        bound = BoundKernel(kernel)
        rebound = bound.rebind(particle_properties={"x": "xidx"})
        assert rebound.particle_property_bindings["x"] == "xidx"
        # original is unchanged
        assert bound.particle_property_bindings["x"] == "x"

    def test_rebind_scalar(self) -> None:
        kernel = ParticleKernel(_noop, scalars=[ScalarDeclaration("_dt", np.float64)])
        bound = BoundKernel(kernel)
        rebound = bound.rebind(scalars={"_dt": "my_dt"})
        assert rebound.scalar_bindings["_dt"] == "my_dt"


# ---------------------------------------------------------------------------
# BoundKernel chaining
# ---------------------------------------------------------------------------


class TestBoundKernelChaining:
    def test_chain_merges_bindings(self) -> None:
        k1 = ParticleKernel(_noop, [ParticlePropertyDeclaration("x", np.float64)])
        k2 = ParticleKernel(_noop2, [ParticlePropertyDeclaration("y", np.float64)])
        b1 = BoundKernel(k1)
        b2 = BoundKernel(k2)
        chained = BoundKernel.chain(b1, b2)
        assert "x" in chained.particle_property_bindings
        assert "y" in chained.particle_property_bindings

    def test_chain_with_method(self) -> None:
        k1 = ParticleKernel(_noop, [ParticlePropertyDeclaration("x", np.float64)])
        k2 = ParticleKernel(_noop2, [ParticlePropertyDeclaration("y", np.float64)])
        b1 = BoundKernel(k1)
        b2 = BoundKernel(k2)
        chained = b1.chain_with(b2)
        assert "x" in chained.particle_property_bindings
        assert "y" in chained.particle_property_bindings


class TestBoundKernelValidation:
    def test_validate_particles_missing_property_raises(self) -> None:
        kernel = ParticleKernel(_noop, [ParticlePropertyDeclaration("x", np.float32)])
        bound = BoundKernel(kernel)
        particles = {"y": np.zeros(3, dtype=np.float32)}
        with pytest.raises(KeyError):
            bound.validate_particles(particles)

    def test_validate_particles_invalid_dtype_raises(self) -> None:
        kernel = ParticleKernel(_noop, [ParticlePropertyDeclaration("x", np.float32)])
        bound = BoundKernel(kernel)
        particles = {"x": np.zeros(3, dtype=np.int32)}
        with pytest.raises(TypeError):
            bound.validate_particles(particles)

    def test_validate_particles_accepts_valid_dtype(self) -> None:
        kernel = ParticleKernel(_noop, [ParticlePropertyDeclaration("x", np.float32)])
        bound = BoundKernel(kernel)
        particles = {"x": np.zeros(3, dtype=np.float32)}
        # should not raise
        bound.validate_particles(particles)
