"""Shared test support for tests/kernels/roms."""

import numpy as np
import numpy.typing as npt
import pytest

from offline_particles.fields import FieldData

NZ = 4
HC = 5.0


# a representative (hc, NZ) pair used across the vertical-coordinate tests
@pytest.fixture
def hc_nz() -> tuple[float, int]:
    return HC, NZ


def _sigma_grid(Nz: int) -> npt.NDArray[np.float64]:
    # vectorized form of _reference.sigma_coordinate, sampled at each integer zidx in [0, Nz)
    return (np.arange(Nz, dtype=np.float64) + 0.5) / Nz - 1.0


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


# h/zeta fields that vary bilinearly in (y, x), so bilinear interpolation is genuinely exercised
@pytest.fixture
def varying_h_zeta_field_data() -> dict[str, FieldData]:
    ii = np.arange(4, dtype=np.float64)[:, None]
    jj = np.arange(4, dtype=np.float64)[None, :]
    h_array = 40.0 + 2.0 * ii + 3.0 * jj
    zeta_array = 0.5 + 0.1 * ii - 0.05 * jj
    return {
        "h": FieldData(h_array, (0.1, -0.15)),
        "zeta": FieldData(zeta_array, (-0.05, 0.2)),
    }


# a stretching function C that is linear in sigma (C = sigma). This degenerates the S-coordinate
# transform to plain sigma coordinates (S == sigma everywhere), acting as a simple sanity
# baseline distinct from the more realistic nonlinear stretching function below.
@pytest.fixture
def linear_C_field_data() -> dict[str, FieldData]:
    C_array = _sigma_grid(NZ)
    return {"C": FieldData(C_array, (0.0,))}


# a strictly-monotonic, nonlinear-in-sigma stretching function, sampled at each rho level
@pytest.fixture
def nonlinear_C_field_data() -> dict[str, FieldData]:
    sigma_grid = _sigma_grid(NZ)
    C_array = sigma_grid + 0.05 * sigma_grid**3
    return {"C": FieldData(C_array, (0.0,))}


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
