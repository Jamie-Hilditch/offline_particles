# Lagrange Polynomial Interpolation

Critical to any Lagrangian particle work is the ability to interpolate fields on to the particle positions. In _Offline Particles_ these fields are defined on structured grids in 1, 2, or 3 dimensions. Interpolation is performed with respect to index space, i.e. the integer lattice enumerating the grid. A set of $n$ values at $n$ nodes uniquely defines a polynomial of degree at most $n-1$. This polynomial may be used to interpolate the values between nodes.

## One-dimensional Interpolation

Consider a $2N$ point stencil of points $\{I + k \mid k = 0, 1, \dots, 2N - 1\}$ with values $v_k$. Suppose the particle is at the fractional index $I + x$. Typically, we choose our $2N$ point stencil such that $N - 1 \leq x < N$ and there are $N$ points to the left of the particle and to the right. However, near the edges of the domain this may not be possible and $x$ may lie outside this range. If the particle lies near the edge of the stencil then the interpolation may be susceptible to Runge's phenomenon.

Let $\xi$ be a continuous index space coordinate with $\xi = k$ on the $k$'th node. The particle is at $\xi = x$.
The $j$'th Lagrange basis polynomial $\ell_j(\xi)$ is a $2N-1$ degree polynomial defined by the property that it is equal to 1 on node $j$ and 0 on the other $2N-1$ nodes. Using the Kronecker delta we can write

$$ \ell_j(k) = \delta_{jk}. $$

Since we know the $2N-1$ roots of the $j$'th basis polynomial we can write down its analytic form

$$ \ell_j(\xi) = \frac{1}{w_j}\prod_{k\neq j} (\xi - k), \quad w_j = \prod_{k\neq j} (j - k). $$

The interpolating polynomial is simply

$$ P(\xi) = \sum_j v_j \ell_j(\xi). $$

## Derivatives in Index Space

We often want to consider the derivative of a field (with respect to some coordinate) interpolated on to the particle position. The building block is the derivative with respect to the index coordinate $\xi$. In other words we're interested in 

$$ P'(\xi) = \sum_j v_j \ell_j'(\xi) $$

where $\ell_j'$ is the derivative of the $j$'th Lagrange basis polynomial. While the basis polynomials themselves have a convenient closed form in terms of their roots, their derivatives are more awkward. Indeed the approach we take is to express them in a monomial basis in which it is trivial to differentiate. It is beneficial to work in a centred coordinate so we define

$$ t = \xi - \left(N - \frac12\right). $$

In typical usage, we evaluate at $t_\star \in [-\frac12, \frac12)$ and hence $t_\star^{2N-1}$ is small. Compare this to $x^{2N-1}$ which gets large very, very quickly. With respect to the centred coordinate the nodes are at

$$ t_k = k - \left(N - \frac12\right) = \frac12 \left(2k - 2N + 1\right). $$

The interpolating polynomial in the centred variable is

$$ \overline{P}(t) = \sum_j v_j\overline{\ell_j}(t). $$

### Vandermonde Matrix Inversion
In the monomial basis

$$ \overline{\ell_j}(t) = \sum_{m} c_{jm} t^m $$

where the coefficient matrix $C = (c_{jm})$ is to be found. To do so we evaluate at the nodes

$$ \overline{\ell_j}(t_k) = \sum_m c_{jm} t_k^m = \delta_{jk}. $$

Introducing the Vandermonde matrix V, with components $V_{km} = t_k^m $, allows us to write 

$$ CV^\top = I \implies C = (V^{-1})^\top. $$

A convenient numerical trick is to note that

$$ V_{km} = t_k^m = \left(2k - 2N + 1\right)^m \left(\frac12\right)^m. $$

Factorising 

$$ V = AS \implies C = \left(S^{-1}A^{-1}\right)^\top $$

where $A = \left(2k - 2N + 1\right)^m$ is an integer matrix and $S = \mathrm{diag}\left(\frac12^0, \frac12^1, \dots\right) \implies S^{-1} = \mathrm{diag}\left(2^0, 2^1, \dots\right)$.

### Derivative Matrix
Taking a derivative in the monomial basis is straight forward

$$ \ell_j'(t) = \sum_n c_{jn}' t^n = \sum_m m c_{jm} t^{m-1} $$

and thus

$$ c'_{jn} = \sum_m c_{jm}d_{mn} \quad \text{with} \quad d_{mn} = \begin{cases} m & \text{if} \quad n = m-1 \\ 0 & \text{otherwise} \end{cases}. $$

Here $D = (d_{mn})$ is the $2N \times 2N$ matrix mapping monomial coefficients to their derivatives. The final column ($n = 2N-1$) of $D$ is identically zero, lowering the polynomial degree by one.

## Derivatives With Respect To Other Fields

Suppose we have two 1D fields, say $u$ and $z$ and we want to find $\frac{\mathrm{d}u}{\mathrm{d}z}$ at the particle location. This is straight-forward using 

$$ \frac{\mathrm{d}u}{\mathrm{d}z} \equiv \frac{\frac{\mathrm{d}u}{\mathrm{d}\xi}}{\frac{\mathrm{d}z}{\mathrm{d}\xi}}. $$

We simply differentiate the two interpolating polynomials, $P_u(\xi)$ and $P_z(\xi)$, and divide

$$ \frac{\mathrm{d}u}{\mathrm{d}z} \approx \frac{P_u'(\xi)}{P_z'(\xi)}. $$

Crucially, the ratio is accurate to the same order as the underlying index space approximations. In this case if $z$ is a coordinate with spacing that scales with $\delta z$ then 

$$ \frac{\mathrm{d}u}{\mathrm{d}z} - \frac{P_u'(\xi)}{P_z'(\xi)} = O\left(\delta z^{2N-1}\right). $$

## Multiple Dimensions

The advantage of interpolating in index space is that in multiple dimensions the index space is simply the Cartesian product of one-dimensional spaces. The Lagrange polynomial basis is simply the product of the 1D basis polynomials. For example in three dimensions we have 

$$ P(\xi, \eta, \zeta) = \sum_{i,j,k} v_{i,j,k} \ell_i(\xi)\ell_j(\eta)\ell_k(\zeta). $$

Derivatives are slightly more work since the physical coordinates don't need to align with the grid. We'll need the Jacobian matrix, i.e. the derivatives of the physical coordinates with respect to indices. If the physical coordinates are $(x,y,z)$ then 

$$ A = \begin{pmatrix}
    \frac{\partial x}{\partial \xi} & \frac{\partial y}{\partial \xi} & \frac{\partial z}{\partial \xi} \\
    \frac{\partial x}{\partial \eta} & \frac{\partial y}{\partial \eta} & \frac{\partial z}{\partial \eta} \\
    \frac{\partial x}{\partial \zeta} & \frac{\partial y}{\partial \zeta} & \frac{\partial z}{\partial \zeta} \\
\end{pmatrix}, 
\quad 
A \begin{pmatrix} 
    \frac{\partial u}{\partial x} \\ 
    \frac{\partial u}{\partial y} \\ 
    \frac{\partial u}{\partial z} 
\end{pmatrix} = \begin{pmatrix}
    \frac{\partial u}{\partial \xi} \\ 
    \frac{\partial u}{\partial \eta} \\ 
    \frac{\partial u}{\partial \zeta} 
\end{pmatrix} \implies \begin{pmatrix} 
    \frac{\partial u}{\partial x} \\ 
    \frac{\partial u}{\partial y} \\ 
    \frac{\partial u}{\partial z} 
\end{pmatrix} = A^{-1} \begin{pmatrix}
    \frac{\partial u}{\partial \xi} \\ 
    \frac{\partial u}{\partial \eta} \\ 
    \frac{\partial u}{\partial \zeta} 
\end{pmatrix}.
$$
For many common grid geometries some of these Jacobian terms are identically zero and so the final expression simplifies. The components of $A$ in addition to the index space derivatives of $u$ can be computed using the above machinary. Alternatively, many models output the metric terms used by the model itself. These can be directly interpolated on to the particle position and used. 

It should be noted that none of the approaches discussed above are guaranteed to preserve any discrete conservation laws used by the model and indeed generally don't. However, this is not normally an important consideration for particle applications.