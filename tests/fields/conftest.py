"""Shared test support for tests/fields."""

import pytest

from offline_particles.spatial_arrays import BBox


@pytest.fixture
def full_domain_bbox():
    def _full_domain_bbox(nx: int) -> BBox:
        return BBox(zmin=0.0, zmax=0.0, ymin=0.0, ymax=0.0, xmin=0.0, xmax=float(nx - 1))

    return _full_domain_bbox
