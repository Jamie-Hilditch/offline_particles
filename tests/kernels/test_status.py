import numpy as np

from offline_particles.kernels.status import Status, is_active, is_inactive

_ACTIVE_MEMBERS = {Status.NORMAL, Status.MULTISTEP_1, Status.MULTISTEP_2}


def test_status_normal_is_zero() -> None:
    assert Status.NORMAL == 0


def test_status_active_inactive_partition_matches_bit_flag_for_scalars() -> None:
    for member in Status:
        expected_active = member in _ACTIVE_MEMBERS
        status = np.uint8(member)
        assert is_active(status) == expected_active
        assert is_inactive(status) == (not expected_active)


def test_status_active_inactive_partition_matches_bit_flag_for_arrays() -> None:
    status = np.array(list(Status), dtype=np.uint8)
    expected_active = np.array([member in _ACTIVE_MEMBERS for member in Status])
    np.testing.assert_array_equal(is_active(status), expected_active)
    np.testing.assert_array_equal(is_inactive(status), ~expected_active)


def test_status_values_fit_in_uint8() -> None:
    for member in Status:
        assert member == np.uint8(member)
