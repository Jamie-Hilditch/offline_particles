# Lagrange Polynomial Interpolation

Critical to any Lagrangian particle work is the ability to interpolate fields on to the particle positions. In _Offline Particles_ these fields are defined on structured grids in 1, 2, or 3 dimensions. Interpolation is performed with respect to index space, i.e. the integer lattice enumerating the grid. A set of $n$ values at $n$ nodes uniquely defines a polynomial of degree at most $n-1$. This polynomial may be used to interpolate the values between nodes.

## One-dimensional interpolation

Consider a $2N$ point stencil of points $\{I + k \mid k = 0, 1, \dots, 2N - 1\}$ with values $v_k$. Suppose the particle is at the fractional index $I + x$. Typically, we choose our $2N$ point stencil such that $N - 1 \leq x < N$ and there are $N$ points to the left of the particle and to the right. However, near the edges of the domain this may not be possible and $x$ may lie outside this range. If the particle lies near the edge of the stencil then the interpolation may be susceptible to Runge's phenomenon.

### Lagrange basis polynomials
It is beneficial to work in a centred coordinate so we define

$$ t = x - \left(N - \frac12\right). $$

With respect to the centred coordinate the nodes are at

$$ t_k = k - \left(N - \frac12\right) = \frac12 \left(2k - 2N + 1\right). $$

The $j$'th Lagrange basis polynomial $\ell_j(t)$ is defined by the property

$$ \ell_j(t_k) = \delta_{jk}, $$

i.e. it is equal to 1 on node $j$ and 0 on all other nodes. The interpolating polynomial is then

$$ P(t) = \sum_{k=0}^{2N-1} v_k\ell_k(t). $$

### Vandermonde inversion
In the monomial basis

$$ \ell_j(t) = \sum_{m=0}^{2N-1} c_{jm} t^m $$

where the coefficient matrix $C = (c_{jm})$ is to be found. To do so we evaluate at the nodes

$$ \ell_j(t_k) = \sum_{m=0}^{2N-1} c_{jm} t_k^m = \delta_{jk}. $$

Introducing the Vandermonde matrix V, with components $V_{km} = t_k^m $, allows us to write 

$$ CV^\top = I \implies C = (V^{-1})^\top. $$

A convenient numerical trick is to note that

$$ V_{km} = t_k^m = \left(2k - 2N + 1\right)^m \left(\frac12\right)^m. $$

Factorising 

$$ V = AS \implies C = \left(S^{-1}A^{-1}\right)^\top $$

where $A = \left(2k - 2N + 1\right)^m$ is an integer matrix and $S = \mathrm{diag}\left(\frac12^0, \frac12^1, \dots\right) \implies S^{-1} = \mathrm{diag}\left(2^0, 2^1, \dots\right)$.

### Derivatives
Taking a derivative in the monomial basis is straight forward

$$ P'(t) = \sum_{k=0}^{2N-1} v_k\ell'_k(t). $$

where

$$ \ell_j'(t) = \sum_{n=0}^{2N-1} c_{jn}' t^n = \sum_{m=0}^{2N-1} m c_{jm} t^{m-1} $$

and thus

$$ c'_{jn} = \sum_{m=0}^{2N-1} c_{jm}d_{mn} \quad \text{with} \quad d_{mn} = \begin{cases} m & \text{if} \quad n = m-1 \\ 0 & \text{otherwise} \end{cases}. $$

Here $D = (d_{mn})$ is the $2N \times 2N$ matrix mapping monomial coefficients to their derivatives. The final column ($n = 2N-1$) of $D$ is identically zero, lowering the polynomial degree by one.
