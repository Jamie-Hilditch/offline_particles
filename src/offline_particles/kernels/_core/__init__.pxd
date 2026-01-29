"""Core utilities for writing kernel functions."""

ctypedef fused prop_t:
    # Supported property types.

    # floats
    float
    double

    # signed ints
    signed char
    short
    int
    long
    long long

    # unsigned ints
    unsigned char
    unsigned short
    unsigned int
    unsigned long
    unsigned long long

ctypedef fused float_t:
    # Supported floating point types.
    float
    double
