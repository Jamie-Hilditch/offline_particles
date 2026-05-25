import numpy as np
import pytest

from offline_particles.kernels import BoundKernel, ParticleKernel, ParticlePropertyDeclaration
from offline_particles.particles import Particles, _find_valid_particle_property_dtype


def _make_bound_kernel(
    declared_name: str,
    dtype_constraint: type[np.generic],
    *,
    bound_name: str | None = None,
) -> BoundKernel:
    kernel = ParticleKernel(lambda pp, sc, fd: None, [ParticlePropertyDeclaration(declared_name, dtype_constraint)])
    if bound_name is None:
        return BoundKernel(kernel)
    return BoundKernel(kernel, particle_property_bindings={declared_name: bound_name})


def test_find_valid_particle_property_dtype_prefers_float64_for_floating_constraint() -> None:
    declarations = [ParticlePropertyDeclaration("x", np.floating)]

    dtype = _find_valid_particle_property_dtype(declarations)

    assert dtype == np.dtype(np.float64)


def test_find_valid_particle_property_dtype_respects_intersection_of_constraints() -> None:
    declarations = [
        ParticlePropertyDeclaration("x", np.floating),
        ParticlePropertyDeclaration("x", np.float32),
    ]

    dtype = _find_valid_particle_property_dtype(declarations)

    assert dtype == np.dtype(np.float32)


def test_find_valid_particle_property_dtype_raises_when_unsatisfiable() -> None:
    declarations = [
        ParticlePropertyDeclaration("x", np.float32),
        ParticlePropertyDeclaration("x", np.int32),
    ]

    with pytest.raises(ValueError, match="No valid dtype found"):
        _find_valid_particle_property_dtype(declarations)


def test_find_valid_particle_property_dtype_supports_datetime64_constraints() -> None:
    declarations = [ParticlePropertyDeclaration("x", np.datetime64)]

    dtype = _find_valid_particle_property_dtype(declarations)

    assert dtype == np.dtype("datetime64[ns]")


def test_find_valid_particle_property_dtype_supports_timedelta64_constraints() -> None:
    declarations = [ParticlePropertyDeclaration("x", np.timedelta64)]

    dtype = _find_valid_particle_property_dtype(declarations)

    assert dtype == np.dtype("timedelta64[ns]")


def test_build_from_kernels_conflicting_constraints_raises() -> None:
    b1 = _make_bound_kernel("x", np.float32)
    b2 = _make_bound_kernel("x", np.int32)
    with pytest.raises(ValueError):
        Particles.build_from_kernels(3, {}, [b1, b2])


def test_build_from_kernels_uses_specified_dtype_and_validates() -> None:
    b = _make_bound_kernel("x", np.float32)
    # valid specified dtype
    p = Particles.build_from_kernels(2, {"x": np.float32}, [b])
    assert p.dtypes["x"] == np.dtype(np.float32)

    # invalid specified dtype should raise
    with pytest.raises(TypeError):
        Particles.build_from_kernels(2, {"x": np.int32}, [b])


def test_build_from_kernels_merges_constraints_by_bound_name() -> None:
    # Different declarations can bind to the same name; constraints should be intersected.
    b1 = _make_bound_kernel("x", np.floating, bound_name="shared")
    b2 = _make_bound_kernel("temperature", np.float32, bound_name="shared")

    particles = Particles.build_from_kernels(4, {}, [b1, b2])

    assert particles.dtypes["shared"] == np.dtype(np.float32)


def test_build_from_kernels_accepts_dtype_like_string_from_user() -> None:
    b = _make_bound_kernel("x", np.floating)

    particles = Particles.build_from_kernels(2, {"x": "float16"}, [b])

    assert particles.dtypes["x"] == np.dtype(np.float16)


def test_build_from_kernels_invalid_dtype_like_string_raises() -> None:
    b = _make_bound_kernel("x", np.floating)

    with pytest.raises(TypeError):
        Particles.build_from_kernels(2, {"x": "not-a-dtype"}, [b])


def test_build_from_kernels_specified_dtype_must_satisfy_all_constraints() -> None:
    b1 = _make_bound_kernel("x", np.floating)
    b2 = _make_bound_kernel("x", np.float32)

    with pytest.raises(TypeError):
        Particles.build_from_kernels(2, {"x": np.float64}, [b1, b2])


def test_build_from_kernels_accepts_user_specified_datetime64_dtype() -> None:
    b = _make_bound_kernel("x", np.datetime64)

    particles = Particles.build_from_kernels(2, {"x": np.dtype("datetime64[ms]")}, [b])

    assert particles.dtypes["x"] == np.dtype("datetime64[ms]")


def test_build_from_kernels_rejects_datetime64_for_timedelta64_constraint() -> None:
    b = _make_bound_kernel("x", np.timedelta64)

    with pytest.raises(TypeError):
        Particles.build_from_kernels(2, {"x": np.dtype("datetime64[ns]")}, [b])
