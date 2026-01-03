"""Particle status codes."""

from .status cimport STATUS

from enum import IntEnum


class ParticleStatus(IntEnum):
    # inactive flag
    INACTIVE = STATUS.INACTIVE

    # normal state
    NORMAL = STATUS.NORMAL

    # error states
    NONFINITE = STATUS.NONFINITE
    OUT_OF_DOMAIN = STATUS.OUT_OF_DOMAIN

    # Reserved for multistep initialization
    MULTISTEP_1 = STATUS.MULTISTEP_1
    MULTISTEP_2 = STATUS.MULTISTEP_2
