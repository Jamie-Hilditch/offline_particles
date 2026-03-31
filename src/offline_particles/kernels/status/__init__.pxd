cdef enum STATUS:
    # bit flag for active/ inactive particles
    INACTIVE = 1 << 7  # reserve final bit for inactive flag

    # normal state
    NORMAL = 0

    # error states
    NONFINITE = 1 | INACTIVE
    OUT_OF_DOMAIN = 2 | INACTIVE
    BELOW_BOTTOM = 3 | INACTIVE
    ABOVE_SURFACE = 4 | INACTIVE

    # Reserved for multistep initialization
    MULTISTEP_1 = 10
    MULTISTEP_2 = 11

    # timed releases and retirements
    PRE_RELEASE = 20 | INACTIVE
    POST_RETIREMENT = 21 | INACTIVE
