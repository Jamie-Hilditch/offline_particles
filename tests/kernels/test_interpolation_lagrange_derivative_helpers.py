"""Direct unit tests for Lagrange derivative basis construction.

These test `_lagrange_derivative_basis_coefficients` and its supporting matrices
(`_factored_vandermonde`, `_inverse_vandermonde`, `_monomial_derivative`) and the codegen'd tuple
evaluator `_lagrange_derivative_basis`, by importing
`offline_particles.kernels.interpolation._lagrange_basis` directly (which now also holds
`_lagrange_basis`, used here for cross-checks against the value-only evaluator), independent of
whether/how the result is exposed publicly. `N`/`derivative_order` validation lives only on the
public `lagrange_basis` wrapper, not on these private internals, and is tested in
`test_interpolation_lagrange_helpers.py`. Tests of the public interface live in
`test_interpolation_lagrange.py`.
"""

from fractions import Fraction

import numpy as np
import pytest
import sympy

from offline_particles.kernels.interpolation._lagrange_basis import (
    _factored_vandermonde,
    _inverse_vandermonde,
    _lagrange_basis,
    _lagrange_derivative_basis,
    _lagrange_derivative_basis_coefficients,
    _monomial_derivative,
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


# standalone exact monomial-coefficient computation via root expansion, written independently
# of production code (no sympy, no matrix inversion) to serve as an oracle for
# _lagrange_derivative_basis_coefficients
def _independent_basis_coefficients(N: int) -> list[list[Fraction]]:
    num_nodes = 2 * N
    nodes = [Fraction(2 * k - 2 * N + 1, 2) for k in range(num_nodes)]
    coefficients = []
    for j in range(num_nodes):
        poly = [Fraction(1)]  # coefficients of prod_{k != j} (t - t_k), low-to-high degree
        for k in range(num_nodes):
            if k == j:
                continue
            new_poly = [Fraction(0)] * (len(poly) + 1)
            for power, coeff in enumerate(poly):
                new_poly[power] += coeff * (-nodes[k])
                new_poly[power + 1] += coeff
            poly = new_poly
        normalization = Fraction(1)
        for k in range(num_nodes):
            if k != j:
                normalization *= nodes[j] - nodes[k]
        coefficients.append([c / normalization for c in poly])
    return coefficients


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_factored_vandermonde_reconstructs_vandermonde_matrix(N: int) -> None:
    n = 2 * N
    A, S = _factored_vandermonde(N)
    nodes = [sympy.Rational(2 * k - (n - 1), 2) for k in range(n)]
    V = sympy.ImmutableMatrix(n, n, lambda k, i: nodes[k] ** i)

    assert A * S == V


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_inverse_vandermonde_is_the_exact_inverse(N: int) -> None:
    n = 2 * N
    A, S = _factored_vandermonde(N)
    V = A * S

    assert V * _inverse_vandermonde(N).T == sympy.eye(n)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_monomial_derivative_matches_symbolic_differentiation(N: int) -> None:
    """D must satisfy phi'(t) = D @ phi(t) for phi(t) = (1, t, ..., t**(2N-1)), proven symbolically."""
    n = 2 * N
    t = sympy.symbols("t")
    phi = sympy.Matrix([t**i for i in range(n)])
    D = _monomial_derivative(N)

    assert (phi.diff(t) - D * phi).applyfunc(sympy.expand) == sympy.zeros(n, 1)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_derivative_basis_coefficients_matches_independent_root_expansion(N: int) -> None:
    coefficients = _lagrange_derivative_basis_coefficients(N, 0)
    oracle = _independent_basis_coefficients(N)

    for j in range(2 * N):
        for i in range(2 * N):
            assert coefficients[j, i] == pytest.approx(float(oracle[j][i]), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_derivative_basis_coefficients_satisfy_kronecker_delta_at_nodes(N: int) -> None:
    coefficients = _lagrange_derivative_basis_coefficients(N, 0)
    num_nodes = 2 * N

    for k in range(num_nodes):
        t_k = k - (N - 0.5)
        for j in range(num_nodes):
            value = sum(coefficients[j, i] * t_k**i for i in range(num_nodes))
            expected = 1.0 if j == k else 0.0
            assert value == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_derivative_basis_coefficients_sum_to_one(N: int) -> None:
    """sum_j l_j(t) == 1 identically, since the basis interpolates the constant function 1 exactly."""
    num_nodes = 2 * N
    coefficients = _lagrange_derivative_basis_coefficients(N, 0)

    expected = np.zeros(num_nodes)
    expected[0] = 1.0
    assert coefficients.sum(axis=0) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_lagrange_derivative_basis_coefficients_mirror_symmetry(N: int, order: int) -> None:
    """Nodes are symmetric about t=0 (t_{2N-1-j} = -t_j), so l_{2N-1-j}^(d)(t) = (-1)**d l_j^(d)(-t)."""
    num_nodes = 2 * N
    coefficients = _lagrange_derivative_basis_coefficients(N, order)
    signs = np.array([(-1) ** (order + i) for i in range(num_nodes)])

    for j in range(num_nodes):
        mirrored = coefficients[num_nodes - 1 - j]
        assert mirrored == pytest.approx(coefficients[j] * signs, abs=1e-9)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_derivative_basis_coefficients_exact_for_degree_2n_minus_1_polynomial(N: int) -> None:
    num_nodes = 2 * N
    degree = num_nodes - 1
    polynomial = np.polynomial.Polynomial(np.arange(1, degree + 2, dtype=np.float64))
    coefficients = _lagrange_derivative_basis_coefficients(N, 0)
    node_values = [polynomial(k - (N - 0.5)) for k in range(num_nodes)]

    for t in [-0.3, 0.0, 0.123, 1.7]:
        interpolated = sum(
            node_values[j] * sum(coefficients[j, i] * t**i for i in range(num_nodes)) for j in range(num_nodes)
        )
        assert interpolated == pytest.approx(polynomial(t), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_derivative_basis_coefficients_matches_independent_product_formula(N: int) -> None:
    coefficients = _lagrange_derivative_basis_coefficients(N, 0)
    num_nodes = 2 * N

    for x in [N - 0.7, N - 0.5, N - 0.2, N + 0.1]:
        t = x - (N - 0.5)
        for j in range(num_nodes):
            expected = _independent_lagrange_basis_weight(x, j, num_nodes)
            actual = sum(coefficients[j, i] * t**i for i in range(num_nodes))
            assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_derivative_basis_coefficients_derivatives_match_numpy_polynomial(N: int) -> None:
    num_nodes = 2 * N
    degree = num_nodes - 1
    polynomial = np.polynomial.Polynomial(np.arange(1, degree + 2, dtype=np.float64))
    node_values = [polynomial(k - (N - 0.5)) for k in range(num_nodes)]

    for order in range(num_nodes):
        coefficients = _lagrange_derivative_basis_coefficients(N, order)
        derivative = polynomial.deriv(m=order)
        for t in [-0.3, 0.0, 0.123, 1.7]:
            interpolated = sum(
                node_values[j] * sum(coefficients[j, i] * t**i for i in range(num_nodes)) for j in range(num_nodes)
            )
            assert interpolated == pytest.approx(derivative(t), rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
@pytest.mark.parametrize("d", [1, 2])
def test_lagrange_derivative_basis_coefficients_derivative_has_zero_high_order_columns(N: int, d: int) -> None:
    num_nodes = 2 * N
    coefficients = _lagrange_derivative_basis_coefficients(N, d)
    assert np.all(coefficients[:, num_nodes - d :] == 0.0)


@pytest.mark.parametrize("N", [1, 2, 3])
def test_lagrange_derivative_basis_coefficients_zero_for_derivative_order_at_or_above_stencil_size(N: int) -> None:
    num_nodes = 2 * N
    assert np.all(_lagrange_derivative_basis_coefficients(N, num_nodes) == 0.0)
    assert np.all(_lagrange_derivative_basis_coefficients(N, num_nodes + 3) == 0.0)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_derivative_basis_coefficients_shape_and_dtype(N: int) -> None:
    coefficients = _lagrange_derivative_basis_coefficients(N, 0)
    assert coefficients.shape == (2 * N, 2 * N)
    assert coefficients.dtype == np.float64


def test_lagrange_derivative_basis_coefficients_is_read_only() -> None:
    coefficients = _lagrange_derivative_basis_coefficients(2, 0)
    assert coefficients.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        coefficients[0, 0] = 1.0


def test_lagrange_derivative_basis_coefficients_is_cached() -> None:
    assert _lagrange_derivative_basis_coefficients(2, 0) is _lagrange_derivative_basis_coefficients(2, 0)
    assert _lagrange_derivative_basis_coefficients(2, 1) is _lagrange_derivative_basis_coefficients(2, 1)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_lagrange_derivative_basis_matches_lagrange_derivative_basis_coefficients(N: int, order: int) -> None:
    num_nodes = 2 * N
    coefficients = _lagrange_derivative_basis_coefficients(N, order)
    derivative_basis = _lagrange_derivative_basis(N, order)

    for x in [N - 0.7, N - 0.5, N - 0.2, N + 0.3]:
        t = x - (N - 0.5)
        values = derivative_basis(x)
        for j in range(num_nodes):
            expected = sum(coefficients[j, i] * t**i for i in range(num_nodes))
            assert values[j] == pytest.approx(expected, rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_lagrange_derivative_basis_order_zero_matches_lagrange_basis(N: int) -> None:
    """derivative_order=0 is explicitly allowed and should reproduce _lagrange_basis's own values.

    Cross-checks two independently-implemented evaluators of the same mathematical function.
    """
    basis = _lagrange_basis(N)
    derivative_basis = _lagrange_derivative_basis(N, 0)
    num_nodes = 2 * N

    for x in [N - 0.7, N - 0.5, N - 0.2, N + 0.3]:
        expected = basis(x)
        actual = derivative_basis(x)
        for j in range(num_nodes):
            assert actual[j] == pytest.approx(expected[j], rel=1e-8, abs=1e-8)


def test_lagrange_derivative_basis_is_cached() -> None:
    assert _lagrange_derivative_basis(2, 1) is _lagrange_derivative_basis(2, 1)
