"""Direct unit tests for the public `lagrange_basis` evaluator construction.

These test `lagrange_basis` (including its `derivative_order` dispatch and validation) and the
shared `_tuplify_basis_evaluation` codegen helper, by importing
`offline_particles.kernels.interpolation._lagrange_basis` directly, independent of whether/how the
result is exposed publicly. Tests of the derivative coefficient construction internals
(`_lagrange_derivative_basis_coefficients`, `_lagrange_derivative_basis`, and their supporting
matrices) live in `test_interpolation_lagrange_derivative_helpers.py`. Tests of the ND kernel
engineering layer (particle/field interpolation factories) live in
`test_interpolation_lagrange.py`.
"""

import numba
import numpy as np
import pytest

from offline_particles.kernels.interpolation._lagrange_basis import (
    _tuplify_basis_evaluation,
    lagrange_basis,
)


# standalone Lagrange basis polynomial evaluator, written independently of production code
# (production used to have an equivalent `_lagrange_basis_polynomial`, but it was removed once
# nothing in src/ used it any more -- this is now the sole, test-local oracle for that formula)
def _independent_lagrange_basis_weight(x: float, j: int, num_nodes: int) -> float:
    weight = 1.0
    for k in range(num_nodes):
        if k == j:
            continue
        weight *= (x - k) / (j - k)
    return weight


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_basis_matches_independent_product_formula(N: int) -> None:
    basis = lagrange_basis(N)
    num_nodes = 2 * N

    for x in [N - 0.7, N - 0.5, N - 0.2, N + 0.1]:
        values = basis(x)
        for j in range(num_nodes):
            assert values[j] == pytest.approx(_independent_lagrange_basis_weight(x, j, num_nodes), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_basis_satisfies_kronecker_delta_at_nodes(N: int) -> None:
    basis = lagrange_basis(N)
    num_nodes = 2 * N

    for k in range(num_nodes):
        values = basis(float(k))
        for j in range(num_nodes):
            expected = 1.0 if j == k else 0.0
            assert values[j] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_basis_sums_to_one(N: int) -> None:
    basis = lagrange_basis(N)

    for x in [0.3, 1.25, N - 0.5, 2 * N - 1.1]:
        assert sum(basis(x)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_basis_exact_for_degree_2n_minus_1_polynomial(N: int) -> None:
    num_nodes = 2 * N
    degree = num_nodes - 1
    polynomial = np.polynomial.Polynomial(np.arange(1, degree + 2, dtype=np.float64))
    basis = lagrange_basis(N)
    node_values = [polynomial(float(k)) for k in range(num_nodes)]

    for x in [0.3, N - 0.5, N + 0.2, 2 * N - 1.3]:
        weights = basis(x)
        interpolated = sum(node_values[j] * weights[j] for j in range(num_nodes))
        assert interpolated == pytest.approx(polynomial(x), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_basis_returns_tuple_of_expected_length(N: int) -> None:
    basis = lagrange_basis(N)
    values = basis(0.5)
    assert isinstance(values, tuple)
    assert len(values) == 2 * N


def test_lagrange_basis_is_cached() -> None:
    assert lagrange_basis(2) is lagrange_basis(2)


@pytest.mark.parametrize("invalid_n", [0, -1])
def test_lagrange_basis_rejects_nonpositive_N(invalid_n: int) -> None:
    with pytest.raises(ValueError, match="N must be a positive integer"):
        lagrange_basis(invalid_n)


@pytest.mark.parametrize("invalid_order", [-1, -5])
def test_lagrange_basis_rejects_negative_derivative_order(invalid_order: int) -> None:
    with pytest.raises(ValueError, match="derivative_order"):
        lagrange_basis(2, invalid_order)


def test_tuplify_basis_evaluation_compiles_expected_tuple() -> None:
    @numba.njit(nogil=True, fastmath=True, inline="always")
    def basis_j(j: int, x: float) -> float:
        return x + j

    evaluator = _tuplify_basis_evaluation(basis_j, N=2)
    assert evaluator(3.0) == (3.0, 4.0, 5.0, 6.0)
