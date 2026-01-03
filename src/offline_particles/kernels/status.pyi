from enum import IntEnum

class ParticleStatus(IntEnum):
    # inactive flag
    INACTIVE: int

    # normal state
    NORMAL: int

    # error states
    NONFINITE: int
    OUT_OF_DOMAIN: int

    # Reserved for multistep initialization
    MULTISTEP_1: int
    MULTISTEP_2: int
