"""Tests for Lagrange interpolation kernel factories."""

import numpy as np
import numpy.typing as npt
import pytest

from offline_particles.kernels.interpolation import (
    lagrange2N_1D_factory,
    lagrange2N_1D_particle_factory,
    lagrange2N_2D_factory,
    lagrange2N_2D_particle_factory,
    lagrange2N_3D_factory,
    lagrange2N_3D_particle_factory,
)


def _stencil_origin_and_local_position(idx: float, N: int, offset: float = 0.0) -> tuple[int, float]:
    offset_idx = idx + offset - (N - 1)
    I0 = int(offset_idx)
    x0 = (N - 1) + offset_idx - I0
    return I0, x0


# standalone Lagrange basis polynomial evaluator, written independently of production code
def _independent_lagrange_basis_weight(x: float, j: int, num_nodes: int) -> float:
    weight = 1.0
    for k in range(num_nodes):
        if k == j:
            continue
        weight *= (x - k) / (j - k)
    return weight


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_1d_exact_for_degree_2n_minus_1_polynomial(N: int) -> None:
    impl = lagrange2N_1D_factory(N=N)
    degree = 2 * N - 1
    polynomial = np.polynomial.Polynomial(np.arange(1, degree + 2, dtype=np.float64))
    idx = np.array([12.375], dtype=np.float64)
    I0, x0 = _stencil_origin_and_local_position(idx[0], N)

    field = polynomial(np.arange(64, dtype=np.float64) - I0)
    status = np.zeros(1, dtype=np.uint8)
    output = np.array([np.nan], dtype=np.float64)

    impl(status, idx, output, field, 0.0)

    expected = polynomial(x0)
    assert output[0] == pytest.approx(expected, rel=1e-11, abs=1e-11)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_1d_not_exact_for_degree_2n_polynomial(N: int) -> None:
    impl = lagrange2N_1D_factory(N=N)
    degree = 2 * N
    polynomial = np.polynomial.Polynomial([0.0] * degree + [1.0])
    idx = np.array([12.375], dtype=np.float64)
    I0, x0 = _stencil_origin_and_local_position(idx[0], N)

    field = polynomial(np.arange(64, dtype=np.float64) - I0)
    status = np.zeros(1, dtype=np.uint8)
    output = np.array([np.nan], dtype=np.float64)

    impl(status, idx, output, field, 0.0)

    expected = polynomial(x0)
    assert abs(output[0] - expected) > 1e-10


def test_lagrange_1d_matches_hand_computed_linear_interpolation() -> None:
    # field = [0, 10, 40, 90]; querying idx=2.3 should bracket indices 2 and 3:
    # 40 + 0.3*(90-40) = 55. A stencil shifted one cell too far left would instead
    # extrapolate from indices 1 and 2, giving 49.
    impl = lagrange2N_1D_particle_factory(N=1)
    field = np.array([0.0, 10.0, 40.0, 90.0], dtype=np.float64)

    result = impl(field, np.float64(2.3), field.shape[0] - 2)

    assert result == pytest.approx(55.0)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_1d_stencil_is_symmetric_about_the_lower_index(N: int) -> None:
    """The 2N-point stencil for idx=k+0.5 must span [k-N+1, k+N]: N points on either side of k."""
    impl = lagrange2N_1D_particle_factory(N=N)
    k = 20
    idx = k + 0.5
    field = np.zeros(64, dtype=np.float64)
    marker_index = k + N  # rightmost point of the correct window; outside the old (shifted) window
    field[marker_index] = 1.0
    max_idx = field.shape[0] - 2 * N

    result = impl(field, np.float64(idx), max_idx)

    x0 = N - 0.5  # local position of idx relative to the correct I0 = k - N + 1
    expected = _independent_lagrange_basis_weight(x0, 2 * N - 1, 2 * N)
    assert expected != 0.0
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("N", [1, 2, 3])
def test_lagrange_2d_stencil_is_symmetric_about_the_lower_index(N: int) -> None:
    impl = lagrange2N_2D_particle_factory(N=N)
    k0, k1 = 20, 18
    idx0, idx1 = k0 + 0.5, k1 + 0.5
    field = np.zeros((64, 64), dtype=np.float64)
    field[k0 + N, k1 + N] = 1.0
    max_idx0 = field.shape[0] - 2 * N
    max_idx1 = field.shape[1] - 2 * N

    result = impl(field, np.float64(idx0), np.float64(idx1), max_idx0, max_idx1)

    weight = _independent_lagrange_basis_weight(N - 0.5, 2 * N - 1, 2 * N)
    expected = weight**2
    assert expected != 0.0
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("N", [1, 2])
def test_lagrange_3d_stencil_is_symmetric_about_the_lower_index(N: int) -> None:
    impl = lagrange2N_3D_particle_factory(N=N)
    k0, k1, k2 = 16, 14, 12
    idx0, idx1, idx2 = k0 + 0.5, k1 + 0.5, k2 + 0.5
    field = np.zeros((40, 40, 40), dtype=np.float64)
    field[k0 + N, k1 + N, k2 + N] = 1.0
    max_idx0 = field.shape[0] - 2 * N
    max_idx1 = field.shape[1] - 2 * N
    max_idx2 = field.shape[2] - 2 * N

    result = impl(field, np.float64(idx0), np.float64(idx1), np.float64(idx2), max_idx0, max_idx1, max_idx2)

    weight = _independent_lagrange_basis_weight(N - 0.5, 2 * N - 1, 2 * N)
    expected = weight**3
    assert expected != 0.0
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("N", [1, 2, 3])
def test_lagrange_2d_exact_for_degree_2n_minus_1_polynomial(N: int) -> None:
    impl = lagrange2N_2D_factory(N=N)
    degree = 2 * N - 1
    idx0 = np.array([14.25], dtype=np.float64)
    idx1 = np.array([16.625], dtype=np.float64)
    I0, local0 = _stencil_origin_and_local_position(idx0[0], N)
    I1, local1 = _stencil_origin_and_local_position(idx1[0], N)

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
    field: npt.NDArray[np.float64] = polynomial(mesh0, mesh1)

    status = np.zeros(1, dtype=np.uint8)
    output = np.array([np.nan], dtype=np.float64)

    impl(status, idx0, idx1, output, field, 0.0, 0.0)

    expected = polynomial(local0, local1)
    assert output[0] == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("N", [1, 2])
def test_lagrange_3d_exact_for_degree_2n_minus_1_polynomial(N: int) -> None:
    impl = lagrange2N_3D_factory(N=N)
    degree = 2 * N - 1
    idx0 = np.array([10.2], dtype=np.float64)
    idx1 = np.array([8.4], dtype=np.float64)
    idx2 = np.array([7.75], dtype=np.float64)
    I0, local0 = _stencil_origin_and_local_position(idx0[0], N)
    I1, local1 = _stencil_origin_and_local_position(idx1[0], N)
    I2, local2 = _stencil_origin_and_local_position(idx2[0], N)

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
    field: npt.NDArray[np.float64] = polynomial(mesh0, mesh1, mesh2)

    status = np.zeros(1, dtype=np.uint8)
    output = np.array([np.nan], dtype=np.float64)

    impl(status, idx0, idx1, idx2, output, field, 0.0, 0.0, 0.0)

    expected = polynomial(local0, local1, local2)
    assert output[0] == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_lagrange_1d_single_particle_matches_parallel_wrapper() -> None:
    parallel_impl = lagrange2N_1D_factory(N=3)
    single_impl = lagrange2N_1D_particle_factory(N=3)

    field = np.linspace(-1.0, 1.0, 64)
    status = np.zeros(1, dtype=np.uint8)
    idx = np.array([12.375], dtype=np.float64)
    output = np.array([np.nan], dtype=np.float64)
    offset = 0.25

    parallel_impl(status, idx, output, field, offset)
    expected = single_impl(field, idx[0] + offset, field.shape[0] - 6)

    assert output[0] == pytest.approx(expected)


def test_lagrange_2d_single_particle_matches_parallel_wrapper() -> None:
    parallel_impl = lagrange2N_2D_factory(N=2)
    single_impl = lagrange2N_2D_particle_factory(N=2)

    field = np.arange(40 * 48, dtype=np.float64).reshape(40, 48)
    status = np.zeros(1, dtype=np.uint8)
    idx0 = np.array([13.375], dtype=np.float64)
    idx1 = np.array([15.625], dtype=np.float64)
    output = np.array([np.nan], dtype=np.float64)
    offset0 = 0.1
    offset1 = -0.2

    parallel_impl(status, idx0, idx1, output, field, offset0, offset1)
    expected = single_impl(field, idx0[0] + offset0, idx1[0] + offset1, field.shape[0] - 4, field.shape[1] - 4)

    assert output[0] == pytest.approx(expected)


def test_lagrange_3d_single_particle_matches_parallel_wrapper() -> None:
    parallel_impl = lagrange2N_3D_factory(N=2)
    single_impl = lagrange2N_3D_particle_factory(N=2)

    field = np.arange(20 * 18 * 16, dtype=np.float64).reshape(20, 18, 16)
    status = np.zeros(1, dtype=np.uint8)
    idx0 = np.array([9.125], dtype=np.float64)
    idx1 = np.array([8.625], dtype=np.float64)
    idx2 = np.array([7.25], dtype=np.float64)
    output = np.array([np.nan], dtype=np.float64)
    offset0 = 0.15
    offset1 = -0.25
    offset2 = 0.05

    parallel_impl(status, idx0, idx1, idx2, output, field, offset0, offset1, offset2)
    expected = single_impl(
        field,
        idx0[0] + offset0,
        idx1[0] + offset1,
        idx2[0] + offset2,
        field.shape[0] - 4,
        field.shape[1] - 4,
        field.shape[2] - 4,
    )

    assert output[0] == pytest.approx(expected)


def test_lagrange_factory_results_are_cached() -> None:
    assert lagrange2N_1D_particle_factory(2) is lagrange2N_1D_particle_factory(2)
    assert lagrange2N_2D_particle_factory(2) is lagrange2N_2D_particle_factory(2)
    assert lagrange2N_3D_particle_factory(2) is lagrange2N_3D_particle_factory(2)
    assert lagrange2N_1D_factory(2, accumulate=True) is lagrange2N_1D_factory(2, accumulate=True)
    assert lagrange2N_2D_factory(2, accumulate=True) is lagrange2N_2D_factory(2, accumulate=True)
    assert lagrange2N_3D_factory(2, accumulate=True) is lagrange2N_3D_factory(2, accumulate=True)


@pytest.mark.parametrize(
    "factory",
    [
        lagrange2N_1D_particle_factory,
        lagrange2N_2D_particle_factory,
        lagrange2N_3D_particle_factory,
        lagrange2N_1D_factory,
        lagrange2N_2D_factory,
        lagrange2N_3D_factory,
    ],
)
@pytest.mark.parametrize("invalid_n", [0, -1])
def test_lagrange_factories_reject_nonpositive_N(factory, invalid_n: int) -> None:
    with pytest.raises(ValueError, match="N must be a positive integer"):
        factory(invalid_n)


def test_lagrange_1d_rejects_field_smaller_than_stencil() -> None:
    impl = lagrange2N_1D_factory(N=2)
    status = np.zeros(1, dtype=np.uint8)
    idx = np.array([0.5], dtype=np.float64)
    output = np.zeros(1, dtype=np.float64)
    field = np.ones(3, dtype=np.float64)

    with pytest.raises(ValueError, match="at least 2N points"):
        impl(status, idx, output, field, 0.0)


def test_lagrange_1d_accepts_field_matching_stencil_size() -> None:
    impl = lagrange2N_1D_factory(N=2)
    status = np.zeros(1, dtype=np.uint8)
    idx = np.array([0.5], dtype=np.float64)
    output = np.zeros(1, dtype=np.float64)
    field = np.ones(4, dtype=np.float64)

    impl(status, idx, output, field, 0.0)


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (3, 3)])
def test_lagrange_2d_rejects_field_smaller_than_stencil(shape: tuple[int, int]) -> None:
    impl = lagrange2N_2D_factory(N=2)
    status = np.zeros(1, dtype=np.uint8)
    idx0 = np.array([0.5], dtype=np.float64)
    idx1 = np.array([0.5], dtype=np.float64)
    output = np.zeros(1, dtype=np.float64)
    field = np.ones(shape, dtype=np.float64)

    with pytest.raises(ValueError, match="at least 2N points in each dimension"):
        impl(status, idx0, idx1, output, field, 0.0, 0.0)


def test_lagrange_2d_accepts_field_matching_stencil_size() -> None:
    impl = lagrange2N_2D_factory(N=2)
    status = np.zeros(1, dtype=np.uint8)
    idx0 = np.array([0.5], dtype=np.float64)
    idx1 = np.array([0.5], dtype=np.float64)
    output = np.zeros(1, dtype=np.float64)
    field = np.ones((4, 4), dtype=np.float64)

    impl(status, idx0, idx1, output, field, 0.0, 0.0)


@pytest.mark.parametrize("shape", [(3, 4, 4), (4, 3, 4), (4, 4, 3), (3, 3, 4)])
def test_lagrange_3d_rejects_field_smaller_than_stencil(shape: tuple[int, int, int]) -> None:
    impl = lagrange2N_3D_factory(N=2)
    status = np.zeros(1, dtype=np.uint8)
    idx0 = np.array([0.5], dtype=np.float64)
    idx1 = np.array([0.5], dtype=np.float64)
    idx2 = np.array([0.5], dtype=np.float64)
    output = np.zeros(1, dtype=np.float64)
    field = np.ones(shape, dtype=np.float64)

    with pytest.raises(ValueError, match="at least 2N points in each dimension"):
        impl(status, idx0, idx1, idx2, output, field, 0.0, 0.0, 0.0)


def test_lagrange_3d_accepts_field_matching_stencil_size() -> None:
    impl = lagrange2N_3D_factory(N=2)
    status = np.zeros(1, dtype=np.uint8)
    idx0 = np.array([0.5], dtype=np.float64)
    idx1 = np.array([0.5], dtype=np.float64)
    idx2 = np.array([0.5], dtype=np.float64)
    output = np.zeros(1, dtype=np.float64)
    field = np.ones((4, 4, 4), dtype=np.float64)

    impl(status, idx0, idx1, idx2, output, field, 0.0, 0.0, 0.0)
