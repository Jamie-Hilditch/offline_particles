"""Submodule for handling vertical coordinate transformations in ROMS.

We follow the same variable naming conventions as ROMS:
- zidx: vertical index
- Nz: total number of vertical rho levels
- hc: critical depth
- sigma: sigma coordinate (uniformly spaced between -1 and 0)
- h: bathymetric depth
- C: stretching function value
- S: S-coordinate value (stretched sigma coordinate)
- zeta: free surface elevation
- z: physical vertical coordinate
"""

cimport cython 

#############################
### computing z from zidx ###
#############################

cdef inline double _sigma_coordinate(double zidx, int Nz) noexcept nogil:
    """Compute the sigma coordinate from a vertical index."""
    return (zidx + 0.5) / Nz - 1.0

cdef inline double _S_coordinate(double hc, double sigma, double h, double C) noexcept nogil:
    """Compute the S-coordinate transformation for ROMS vertical coordinates."""
    return (hc * sigma + h * C) / (hc + h)

cdef inline double _z_coordinate(double S, double h, double zeta) noexcept nogil:
    """Convert S-coordinate to physical coordinate."""
    return zeta + (zeta + h) * S

cdef inline double compute_z(double zidx, int Nz, double hc, double h, double C, double zeta) noexcept nogil:
    """Compute the physical vertical coordinate from the vertical index."""
    cdef double sigma = _sigma_coordinate(zidx, Nz)
    cdef double S = _S_coordinate(hc, sigma, h, C)
    return _z_coordinate(S, h, zeta)

#############################
### computing zidx from z ###  
#############################

cdef inline double _S_from_z(double z, double h, double zeta) noexcept nogil:
    """Convert physical coordinate to S-coordinate."""
    return (z - zeta) / (zeta + h)

cdef inline double _sigma_from_Cidx(int Cidx, double C_offset, int Nz) noexcept nogil:
    """Compute the sigma coordinate from a stretching function index."""
    cdef double zidx = Cidx - C_offset
    return _sigma_coordinate(zidx, Nz)

cdef inline Py_ssize_t _compute_Cidx_from_S(double S, double hc, int NZ, double h, double zeta, const double[::1] C, double C_offset) noexcept nogil:
    """Compute the C array index from the S coordinate.
    
    C_idx corresponds to the index in the stretching function array C such that
        S_coordinate(hc, sigma_from_Cidx(C_idx, C_offset, NZ), h, C[C_idx]) <= S
    and 
        S_coordinate(hc, sigma_from_Cidx(C_idx + 1, C_offset, NZ), h, C[C_idx + 1]) > S
    This is done via a binary search over the C array.
    If S is outside the range of S values defined by C, the first or penultimate index is returned.
    """
    cdef Py_ssize_t C_size = C.shape[0]
    cdef Py_ssize_t lo = 0
    cdef Py_ssize_t hi = C_size - 2
    cdef Py_ssize_t mid
    cdef double S_mid

    # Handle edge cases where S is outside the range of S values
    if S <= _S_coordinate(hc, _sigma_from_Cidx(0, C_offset, NZ), h, C[0]):
        return 0
    elif S >= _S_coordinate(hc, _sigma_from_Cidx(C_size - 2, C_offset, NZ), h, C[C_size - 2]):
        return C_size - 2

    # Binary search
    while lo < hi - 1:
        mid = (lo + hi) // 2
        S_mid = _S_coordinate(hc, _sigma_from_Cidx(mid, C_offset, NZ), h, C[mid])
        
        if S_mid <= S:
            lo = mid
        else:
            hi = mid

    return lo

cdef inline double _zidx_from_S(double S, double hc, int NZ, double h, double zeta, const double[::1] C, double C_offset) noexcept nogil:
    """Compute the vertical index from the S coordinate."""
    # Find the C index corresponding to S
    cdef Py_ssize_t Cidx = _compute_Cidx_from_S(S, hc, NZ, h, zeta, C, C_offset)    
    # Linear interpolation to find the fractional index
    cdef double S_low = _S_coordinate(hc, _sigma_from_Cidx(Cidx, C_offset, NZ), h, C[Cidx])
    cdef double S_high = _S_coordinate(hc, _sigma_from_Cidx(Cidx + 1, C_offset, NZ), h, C[Cidx + 1])
    cdef double f = (S - S_low) / (S_high - S_low)
    return Cidx - C_offset + f

cdef inline double compute_zidx(double z, double h, double zeta, double hc, int NZ, const double[::1] C, double C_offset) noexcept nogil:
    """Compute the vertical index from the physical vertical coordinate."""
    cdef double S = _S_from_z(z, h, zeta)
    return _zidx_from_S(S, hc, NZ, h, zeta, C, C_offset)