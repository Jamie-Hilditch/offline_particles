"""Tests for the FieldData dataclass."""

import numpy as np

from offline_particles.fields import FieldData


class TestFieldDataUnpack:
    def test_unpack_returns_array_followed_by_offsets(self) -> None:
        array = np.array([1.0, 2.0, 3.0])
        field_data = FieldData(array=array, offsets=(0.5, -0.5))

        unpacked = field_data.unpack()

        assert unpacked[0] is array
        assert unpacked[1:] == (0.5, -0.5)

    def test_unpack_with_no_offsets(self) -> None:
        array = np.array([1.0])
        field_data = FieldData(array=array, offsets=())

        unpacked = field_data.unpack()

        assert len(unpacked) == 1
        assert unpacked[0] is array
