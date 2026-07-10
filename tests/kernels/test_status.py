import numpy as np

from offline_particles.kernels.status import Status, is_active, is_inactive

_ACTIVE_MEMBERS = {Status.NORMAL, Status.MULTISTEP_1, Status.MULTISTEP_2}


def test_status_normal_is_zero() -> None:
    assert Status.NORMAL == 0


def test_status_active_inactive_partition_matches_bit_flag() -> None:
    for member in Status:
        expected_active = member in _ACTIVE_MEMBERS
        status = np.array([member], dtype=np.uint8)
        assert bool(is_active(status)[0]) == expected_active
        assert bool(is_inactive(status)[0]) == (not expected_active)


def test_status_values_fit_in_uint8() -> None:
    for member in Status:
        assert member == np.uint8(member)
