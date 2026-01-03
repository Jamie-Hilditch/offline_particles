cdef enum STATUS:
    # bit flag for active/ inactive particles
    INACTIVE = 1 << 7  # reserve final bit for inactive flag

    # normal state
    NORMAL = 0

    # error states
    NONFINITE = 1 | INACTIVE
    OUT_OF_DOMAIN = 2 | INACTIVE

    # Reserved for multistep initialization
    MULTISTEP_1 = 10
    MULTISTEP_2 = 11
