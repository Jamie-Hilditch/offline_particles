"""Tests for the public single-particle Lagrange interpolation factories.

These exercise `lagrange_{1,2,3}D_particle_factory` as exported from
`offline_particles.kernels.interpolation`. The vectorized/field factories that used to live
alongside them are now private, kernel-function-returning implementation details consumed only by
`_kernel_constructors.py`; their coverage (exactness through the full construct_*_interpolation_kernel
pipeline, field-size validation, etc.) lives in `test_interpolation_linear.py` instead. Tests of
private construction details (e.g. `lagrange_basis` and its derivative dispatch) live in
`test_interpolation_lagrange_helpers.py`/`test_interpolation_lagrange_derivative_helpers.py`,
importing directly from `._lagrange_basis`.
"""

import numpy as np
import numpy.typing as npt
import pytest

from offline_particles.kernels.interpolation import (
    lagrange_1D_particle_factory,
    lagrange_2D_particle_factory,
    lagrange_3D_particle_factory,
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


def test_lagrange_1d_matches_hand_computed_linear_interpolation() -> None:
    # field = [0, 10, 40, 90]; querying idx=2.3 should bracket indices 2 and 3:
    # 40 + 0.3*(90-40) = 55. A stencil shifted one cell too far left would instead
    # extrapolate from indices 1 and 2, giving 49.
    impl = lagrange_1D_particle_factory(N=1)
    field = np.array([0.0, 10.0, 40.0, 90.0], dtype=np.float64)

    result = impl(field, np.float64(2.3), field.shape[0] - 2)

    assert result == pytest.approx(55.0)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_1d_stencil_is_symmetric_about_the_lower_index(N: int) -> None:
    """The 2N-point stencil for idx=k+0.5 must span [k-N+1, k+N]: N points on either side of k."""
    impl = lagrange_1D_particle_factory(N=N)
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


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_2d_stencil_is_symmetric_about_the_lower_index(N: int) -> None:
    impl = lagrange_2D_particle_factory(N, N)
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


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_3d_stencil_is_symmetric_about_the_lower_index(N: int) -> None:
    impl = lagrange_3D_particle_factory(N, N, N)
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


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_1d_particle_exact_for_degree_2n_minus_1_polynomial(N: int) -> None:
    impl = lagrange_1D_particle_factory(N)
    degree = 2 * N - 1
    polynomial = np.polynomial.Polynomial(np.arange(1, degree + 2, dtype=np.float64))
    idx = np.float64(12.375)
    I0, x0 = _stencil_origin_and_local_position(idx, N)
    field = polynomial(np.arange(64, dtype=np.float64) - I0)
    max_idx = field.shape[0] - 2 * N

    result = impl(field, idx, max_idx)

    assert result == pytest.approx(polynomial(x0), rel=1e-11, abs=1e-11)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_1d_particle_not_exact_for_degree_2n_polynomial(N: int) -> None:
    impl = lagrange_1D_particle_factory(N)
    degree = 2 * N
    polynomial = np.polynomial.Polynomial([0.0] * degree + [1.0])
    idx = np.float64(12.375)
    I0, x0 = _stencil_origin_and_local_position(idx, N)
    field = polynomial(np.arange(64, dtype=np.float64) - I0)
    max_idx = field.shape[0] - 2 * N

    result = impl(field, idx, max_idx)

    assert abs(result - polynomial(x0)) > 1e-10


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_2d_particle_exact_for_degree_2n_minus_1_polynomial(N: int) -> None:
    impl = lagrange_2D_particle_factory(N, N)
    degree = 2 * N - 1
    idx0, idx1 = np.float64(14.25), np.float64(16.625)
    I0, local0 = _stencil_origin_and_local_position(idx0, N)
    I1, local1 = _stencil_origin_and_local_position(idx1, N)

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
    max_idx0 = field.shape[0] - 2 * N
    max_idx1 = field.shape[1] - 2 * N

    result = impl(field, idx0, idx1, max_idx0, max_idx1)

    assert result == pytest.approx(polynomial(local0, local1), rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_3d_particle_exact_for_degree_2n_minus_1_polynomial(N: int) -> None:
    impl = lagrange_3D_particle_factory(N, N, N)
    degree = 2 * N - 1
    idx0, idx1, idx2 = np.float64(10.2), np.float64(8.4), np.float64(7.75)
    I0, local0 = _stencil_origin_and_local_position(idx0, N)
    I1, local1 = _stencil_origin_and_local_position(idx1, N)
    I2, local2 = _stencil_origin_and_local_position(idx2, N)

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
    max_idx0 = field.shape[0] - 2 * N
    max_idx1 = field.shape[1] - 2 * N
    max_idx2 = field.shape[2] - 2 * N

    result = impl(field, idx0, idx1, idx2, max_idx0, max_idx1, max_idx2)

    assert result == pytest.approx(polynomial(local0, local1, local2), rel=1e-10, abs=1e-10)


@pytest.mark.parametrize(("N0", "N1"), [(1, 3), (3, 1)])
def test_lagrange_2d_particle_axis_0_exactness_depends_only_on_N0(N0: int, N1: int) -> None:
    """A field that only varies along axis 0 isolates axis 0's own stencil capacity.

    Parametrizing both (N0, N1) orderings catches an axis mixup: if axis 0 accidentally used N1's
    evaluator, this would only fail for the (3, 1) case below where N1 < N0.
    """
    impl = lagrange_2D_particle_factory(N0, N1)
    degree = 2 * N0 - 1
    polynomial = np.polynomial.Polynomial(np.arange(1, degree + 2, dtype=np.float64))
    idx0, idx1 = np.float64(14.25), np.float64(16.625)
    I0, local0 = _stencil_origin_and_local_position(idx0, N0)
    n0, n1 = 2 * N0, 2 * N1

    field = np.tile(polynomial(np.arange(n0 + 20, dtype=np.float64) - I0)[:, None], (1, n1 + 20))
    max_idx0 = field.shape[0] - n0
    max_idx1 = field.shape[1] - n1

    result = impl(field, idx0, idx1, max_idx0, max_idx1)

    assert result == pytest.approx(polynomial(local0), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(("N0", "N1"), [(1, 3), (3, 1)])
def test_lagrange_2d_particle_axis_0_not_exact_beyond_N0_capacity(N0: int, N1: int) -> None:
    """Complements the test above: a degree-2N0 term in axis 0 must NOT be exact for axis 0's own N0.

    If axis 0 were accidentally using N1's evaluator, this would wrongly come out exact whenever
    N1 > N0 (the (1, 3) case), catching the mixup from the other direction.
    """
    impl = lagrange_2D_particle_factory(N0, N1)
    degree = 2 * N0  # one degree too high for axis 0's own stencil
    polynomial = np.polynomial.Polynomial([0.0] * degree + [1.0])
    idx0, idx1 = np.float64(14.25), np.float64(16.625)
    I0, local0 = _stencil_origin_and_local_position(idx0, N0)
    n0, n1 = 2 * N0, 2 * N1

    field = np.tile(polynomial(np.arange(n0 + 20, dtype=np.float64) - I0)[:, None], (1, n1 + 20))
    max_idx0 = field.shape[0] - n0
    max_idx1 = field.shape[1] - n1

    result = impl(field, idx0, idx1, max_idx0, max_idx1)

    assert abs(result - polynomial(local0)) > 1e-8


def test_lagrange_2d_particle_mixed_derivative_order_differentiates_only_the_requested_axis() -> None:
    N0, N1 = 2, 2
    impl = lagrange_2D_particle_factory(N0, N1, D0=1, D1=0)
    degree0, degree1 = 2 * N0 - 1, 2 * N1 - 1
    p0 = np.polynomial.Polynomial(np.arange(1, degree0 + 2, dtype=np.float64))
    p1 = np.polynomial.Polynomial(np.arange(1, degree1 + 2, dtype=np.float64))
    dp0 = p0.deriv(1)

    idx0, idx1 = np.float64(10.25), np.float64(8.625)
    I0, local0 = _stencil_origin_and_local_position(idx0, N0)
    I1, local1 = _stencil_origin_and_local_position(idx1, N1)
    n0, n1 = 2 * N0, 2 * N1

    grid0 = np.arange(n0 + 20, dtype=np.float64) - I0
    grid1 = np.arange(n1 + 20, dtype=np.float64) - I1
    mesh0, mesh1 = np.meshgrid(grid0, grid1, indexing="ij")
    field: npt.NDArray[np.float64] = p0(mesh0) * p1(mesh1)
    max_idx0 = field.shape[0] - n0
    max_idx1 = field.shape[1] - n1

    result = impl(field, idx0, idx1, max_idx0, max_idx1)

    expected = dp0(local0) * p1(local1)
    assert result == pytest.approx(expected, rel=1e-8, abs=1e-8)


def test_lagrange_particle_factory_results_are_cached() -> None:
    assert lagrange_1D_particle_factory(2) is lagrange_1D_particle_factory(2)
    assert lagrange_2D_particle_factory(2, 2) is lagrange_2D_particle_factory(2, 2)
    assert lagrange_3D_particle_factory(2, 2, 2) is lagrange_3D_particle_factory(2, 2, 2)


def test_lagrange_1d_particle_factory_rejects_nonpositive_N() -> None:
    for invalid_n in (0, -1):
        with pytest.raises(ValueError, match="N must be a positive integer"):
            lagrange_1D_particle_factory(invalid_n)


@pytest.mark.parametrize("invalid_n", [0, -1])
@pytest.mark.parametrize("axis", [0, 1])
def test_lagrange_2d_particle_factory_rejects_nonpositive_N_on_either_axis(invalid_n: int, axis: int) -> None:
    N0, N1 = (invalid_n, 2) if axis == 0 else (2, invalid_n)
    with pytest.raises(ValueError, match="N must be a positive integer"):
        lagrange_2D_particle_factory(N0, N1)


@pytest.mark.parametrize("invalid_n", [0, -1])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_lagrange_3d_particle_factory_rejects_nonpositive_N_on_any_axis(invalid_n: int, axis: int) -> None:
    Ns = [2, 2, 2]
    Ns[axis] = invalid_n
    with pytest.raises(ValueError, match="N must be a positive integer"):
        lagrange_3D_particle_factory(*Ns)


@pytest.mark.parametrize("invalid_d", [-1, -5])
def test_lagrange_1d_particle_factory_rejects_negative_derivative_order(invalid_d: int) -> None:
    with pytest.raises(ValueError, match="derivative_order"):
        lagrange_1D_particle_factory(2, invalid_d)


@pytest.mark.parametrize("axis", [0, 1])
def test_lagrange_2d_particle_factory_rejects_negative_derivative_order_on_either_axis(axis: int) -> None:
    D0, D1 = (-1, 0) if axis == 0 else (0, -1)
    with pytest.raises(ValueError, match="derivative_order"):
        lagrange_2D_particle_factory(2, 2, D0, D1)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_lagrange_3d_particle_factory_rejects_negative_derivative_order_on_any_axis(axis: int) -> None:
    Ds = [0, 0, 0]
    Ds[axis] = -1
    with pytest.raises(ValueError, match="derivative_order"):
        lagrange_3D_particle_factory(2, 2, 2, *Ds)
