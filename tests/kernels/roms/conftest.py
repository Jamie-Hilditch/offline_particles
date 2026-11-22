"""Shared test support for tests/kernels/roms."""

import numpy as np
import pytest

from offline_particles.fields import FieldData
from offline_particles.kernels.roms import LinearVerticalTransform2

NZ = 4
HC = 5.0


# a representative (hc, NZ) pair used across the vertical-coordinate tests
@pytest.fixture
def hc_nz() -> tuple[float, int]:
    return HC, NZ


# a representative LinearVerticalTransform2 (linear-in-sigma Cs_w, so Cs_w == sigma_w exactly,
# degenerating the S-coordinate transform to plain sigma coordinates) for wiring-level tests that
# don't care about the specifics of the vertical-coordinate math
@pytest.fixture
def linear_vertical_transform(hc_nz: tuple[float, int]) -> LinearVerticalTransform2:
    hc, NZ = hc_nz
    Cs_w = np.arange(NZ + 1, dtype=np.float64) / NZ - 1.0
    return LinearVerticalTransform2(NZ=NZ, hc=hc, Cs_w=Cs_w)


# uniform h/zeta fields, so horizontal bilinear interpolation is trivial to reason about;
# field offsets are nonzero (but small) so the offset-handling plumbing is still exercised
@pytest.fixture
def uniform_h_zeta_field_data() -> dict[str, FieldData]:
    h_array = np.full((4, 4), 50.0, dtype=np.float64)
    zeta_array = np.full((4, 4), 0.5, dtype=np.float64)
    return {
        "h": FieldData(h_array, (0.3, -0.2)),
        "zeta": FieldData(zeta_array, (-0.1, 0.15)),
    }


# a builder for the particle-properties dict shared by both vertical-coordinate kernels
@pytest.fixture
def make_particle_properties():
    def _make_particle_properties(
        *,
        zidx,
        yidx,
        xidx,
        z,
        status=None,
    ) -> dict[str, np.ndarray]:
        zidx = np.asarray(zidx, dtype=np.float64)
        yidx = np.asarray(yidx, dtype=np.float64)
        xidx = np.asarray(xidx, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        n = zidx.shape[0]
        status = np.zeros(n, dtype=np.uint8) if status is None else np.asarray(status, dtype=np.uint8)
        return {"status": status, "zidx": zidx, "yidx": yidx, "xidx": xidx, "z": z}

    return _make_particle_properties
