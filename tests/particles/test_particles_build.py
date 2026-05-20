import numpy as np
import pytest

from offline_particles.kernels import BoundKernel, ParticleKernel, ParticlePropertyDeclaration
from offline_particles.particles import Particles


def _make_bound_kernel_for_x(dtype):
    k = ParticleKernel(lambda pp, sc, fd: None, [ParticlePropertyDeclaration("x", dtype)])
    return BoundKernel(k)


def test_build_from_kernels_conflicting_constraints_raises() -> None:
    b1 = _make_bound_kernel_for_x(np.float32)
    b2 = _make_bound_kernel_for_x(np.int32)
    with pytest.raises(ValueError):
        Particles.build_from_kernels(3, {}, [b1, b2])


def test_build_from_kernels_uses_specified_dtype_and_validates() -> None:
    b = _make_bound_kernel_for_x(np.float32)
    # valid specified dtype
    p = Particles.build_from_kernels(2, {"x": np.float32}, [b])
    assert p.dtypes["x"] == np.dtype(np.float32)

    # invalid specified dtype should raise
    with pytest.raises(TypeError):
        Particles.build_from_kernels(2, {"x": np.int32}, [b])
