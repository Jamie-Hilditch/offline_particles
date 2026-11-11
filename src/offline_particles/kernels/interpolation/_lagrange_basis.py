"""Implementations of Lagrange polynomial bases and their derivatives."""

import functools
from collections.abc import Callable

import numba
import numpy as np
import numpy.typing as npt
import sympy


def _tuplify_basis_evaluation(basis_j: Callable[[int, float], float], N: int) -> Callable[[float], tuple[float, ...]]:
    r"""Create a function that returns a 2N-tuple of basis functions evaluated at :math:`x`.

    Parameters
    ----------
    basis_j : Callable[[int, float], float]
        A numba jitted function that evaluates the basis polynomial, taking the basis index :math:`j` and coordinate :math:`x`.
    N : int
        The half-width of the stencil.

    Returns
    -------
    Callable[[float], tuple[float, ...]]
        A numba jitted function taking the coordinate :math:`x \in [0, 2N-1)` and returning
        a tuple of the 2N basis values at :math:`x`.

    Notes
    -----
    We use code generation with :func:`exec` here in order to return a tuple of length 2N. This avoids an array allocation
    that numba cannot otherwise optimise away.
    """
    n = 2 * N
    namespace = {"basis_j": basis_j}

    # build up output tuple
    basis_tuple = "("
    for j in range(n):
        basis_tuple += f"basis_j({j}, x), "
    basis_tuple += ")"

    # generate function with exec
    codegen = f"def basis_evaluation(x):\n    return {basis_tuple}\n"
    exec(codegen, namespace)  # noqa: S102 [exec use]
    basis_evaluation = namespace["basis_evaluation"]

    return numba.njit(nogil=True, fastmath=True, inline="always")(basis_evaluation)


@functools.cache
def _lagrange_basis(N: int) -> Callable[[float], tuple[float, ...]]:
    r"""Create a function evaluating the 2N Lagrange basis polynomials at :math:`x`.

    Parameters
    ----------
    N : int
        The half width of the stencil. The basis has 2N functions of degree 2N - 1.

    Returns
    -------
    Callable[[float], tuple[float, ...]]
        A numba jitted function taking the coordinate :math:`x \in [0, 2N-1)` and returning
        a tuple of the 2N basis values at :math:`x`.
    """
    n = 2 * N
    weights = [np.prod([j - k for k in range(n) if k != j]) for j in range(n)]
    factors = 1 / np.array(weights, dtype=np.float64)

    @numba.njit(nogil=True, fastmath=True, inline="always")
    def basis_j(j: int, x: float) -> float:
        lbp = factors[j]
        for k in range(n):
            if j == k:
                continue
            lbp *= x - k
        return lbp

    return _tuplify_basis_evaluation(basis_j, N)


def _factored_vandermonde(N: int) -> tuple[sympy.ImmutableMatrix, sympy.ImmutableMatrix]:
    r"""Construct the factors of the Vandermonde matrix on the 2N centred nodes.

    The centred nodes are

    .. math::

        t_k = \frac{\alpha_k}{2}, \quad \alpha_k = 2k - (2N - 1), \quad k = 0, \ldots, 2N - 1.

    The Vandermonde matrix :math:`V_{k, i} = t_k^i` factors as :math:`V = A S`.

    Parameters
    ----------
    N : int
        The half width of the stencil. The matrices are 2N x 2N.

    Returns
    -------
    A : sympy.ImmutableMatrix
        The integer Vandermonde matrix :math:`A_{k, i} = \alpha_k^i`.
    S : sympy.ImmutableMatrix
        The diagonal matrix :math:`S = \mathrm{diag}(2^{-i})` of powers of two.
    """
    n = 2 * N
    alpha = [2 * k - (n - 1) for k in range(n)]
    half_powers = [sympy.Rational(1, 2**i) for i in range(n)]
    A = sympy.ImmutableMatrix(n, n, lambda m, i: alpha[m] ** i)
    S = sympy.ImmutableMatrix(sympy.diag(*half_powers))
    return A, S


def _monomial_derivative(N: int) -> sympy.ImmutableMatrix:
    r"""Construct the derivative matrix in the monomial basis of degree 2N - 1.

    With :math:`\phi(t) = (1, t, \ldots, t^{2N - 1})`, the matrix satisfies :math:`\phi'(t) = D \phi(t)`,
    so its only nonzero entries are :math:`D_{i, i - 1} = i`.

    Parameters
    ----------
    N : int
        The half width of the stencil. The matrix is 2N x 2N.

    Returns
    -------
    sympy.ImmutableMatrix
        The integer derivative matrix :math:`D`.
    """
    n = 2 * N
    return sympy.ImmutableMatrix(n, n, lambda i, k: i if i == k + 1 else 0)


def _inverse_vandermonde(N: int) -> sympy.ImmutableMatrix:
    """Construct the inverse of the Vandermonde matrix.

    Parameters
    ----------
    N : int
        The half width of the stencil. The matrices are 2N x 2N.

    Returns
    -------
    sympy.ImmutableMatrix
        The inverse of the Vandermonde matrix.
    """
    A, S = _factored_vandermonde(N)
    V_inv = A.inv().T * S.inv()
    return sympy.ImmutableMatrix(V_inv)


@functools.cache
def _lagrange_derivative_basis_coefficients(N: int, derivative_order: int) -> npt.NDArray[np.float64]:
    r"""Monomial coefficients of the Lagrange basis on the 2N centred nodes.

    With :math:`V = A S` from :func:`_factored_vandermonde`, the coefficients are

    .. math::

        C = V^{-T} = A^{-T} S^{-1},

    so that the basis values are :math:`\ell(t) = C \phi(t)` in the centred variable
    :math:`t = x - (N - \tfrac{1}{2})`. The :math:`d`-th derivative coefficients are :math:`CD^d`, with :math:`D`
    from :func:`_monomial_derivative`. All arithmetic is exact; the result is rounded to float64 once.

    Parameters
    ----------
    N : int
        The half width of the stencil. The basis has 2N functions of degree 2N - 1.
    derivative_order : int
        The order :math:`d` of the derivative with respect to :math:`t` (or equivalently :math:`x`).

    Returns
    -------
    npt.NDArray[np.float64]
        Read-only array of shape (2N, 2N) where entry :math:`[j, i]` is the coefficient of :math:`t^i`
        in the :math:`d`-th derivative of :math:`\ell_j`. The last :math:`d` columns are zero.
    """
    C = _inverse_vandermonde(N)
    D = _monomial_derivative(N)
    C = C * D**derivative_order  # exact derivative coefficients

    out = sympy.matrices.matrix2numpy(C, dtype=np.float64)
    out.flags.writeable = False
    return out


@functools.cache
def _lagrange_derivative_basis(N: int, derivative_order: int) -> Callable[[float], tuple[float, ...]]:
    r"""Create a function evaluating the 2N Lagrange derivative basis polynomials at :math:`x`.

    Parameters
    ----------
    N : int
        The half width of the stencil. The basis has 2N functions of degree 2N - 1.
    derivative_order : int
        The order :math:`d` of the derivative with respect to :math:`x`.

    Returns
    -------
    Callable[[float], tuple[float, ...]]
        A numba jitted function taking the coordinate :math:`x` and returning
        a tuple of the 2N :math:`d`-th basis derivatives at :math:`x`.
    """
    n = 2 * N
    C = _lagrange_derivative_basis_coefficients(N, derivative_order)

    @numba.njit(nogil=True, fastmath=True, inline="always")
    def basis_j(j: int, x: float) -> float:
        t = x - (N - 0.5)
        acc = 0.0
        for k in range(n - 1, -1, -1):  # Horner in t
            acc = acc * t + C[j, k]
        return acc

    return _tuplify_basis_evaluation(basis_j, N)


@functools.cache
def lagrange_basis(N: int, derivative_order: int = 0) -> Callable[[float], tuple[float, ...]]:
    r"""Create a function evaluating the 2N Lagrange basis polynomials or their derivatives at :math:`x`.

    Parameters
    ----------
    N : int
        The half width of the stencil. The basis has 2N functions of degree 2N - 1.
    derivative_order : int, optional
        The order :math:`d` of the derivative with respect to :math:`x`.

    Returns
    -------
    Callable[[float], tuple[float, ...]]
        A numba jitted function taking the coordinate :math:`x` and returning
        a tuple of the 2N :math:`d`-th basis derivatives at :math:`x`.

    Raises
    ------
    ValueError
        If `N` is not positive or `derivative_order` is not a non-negative integer.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    match derivative_order:
        case 0:
            return _lagrange_basis(N)
        case int() as D if D > 0:
            return _lagrange_derivative_basis(N, D)
        case _:
            raise ValueError("Invalid value for derivative_order. Expected non-negative integer.")
