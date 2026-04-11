"""Tests for Fieldset.from_xarray."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from offline_particles.fields import StaticField, TimeDependentField
from offline_particles.fieldset import Fieldset
from offline_particles.spatial_arrays import Dimension, Stagger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T, NZ, NY, NX = 3, 5, 6, 7


def _make_dataset() -> xr.Dataset:
    """Dataset with a typical set of ROMS-like variables."""
    return xr.Dataset(
        {
            # time-dependent, full 3-D
            "u": xr.DataArray(np.ones((T, NZ, NY, NX), dtype=np.float64), dims=["time", "z", "y", "x"]),
            # static, 2-D
            "h": xr.DataArray(np.ones((NY, NX), dtype=np.float64), dims=["y", "x"]),
            # static, 1-D in z only
            "C": xr.DataArray(np.ones((NZ,), dtype=np.float64), dims=["z"]),
        }
    )


DIM_MAP = {
    "time": Dimension.TIME,
    "z": Dimension.Z_CENTER,
    "y": Dimension.Y_CENTER,
    "x": Dimension.X_CENTER,
}


# ---------------------------------------------------------------------------
# Size inference
# ---------------------------------------------------------------------------


class TestFromXarraySizeInference:
    def test_infers_all_sizes_from_dataset(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, DIM_MAP)
        assert fs.t_size == T
        assert fs.z_size == NZ
        assert fs.y_size == NY
        assert fs.x_size == NX

    def test_explicit_sizes_override_inferred(self) -> None:
        # Explicit sizes that match the actual data should be accepted even when
        # the dataset would infer the same values – verifying the override path.
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, DIM_MAP, t_size=T, z_size=NZ, y_size=NY, x_size=NX)
        assert fs.t_size == T
        assert fs.z_size == NZ
        assert fs.y_size == NY
        assert fs.x_size == NX

    def test_infers_sizes_from_staggered_dims(self) -> None:
        # xi_u has NX-1 points (INNER) – z_size should still be inferred as NX
        ds = xr.Dataset(
            {
                "u": xr.DataArray(
                    np.ones((T, NZ, NY, NX - 1), dtype=np.float64),
                    dims=["time", "z", "y", "xi_u"],
                ),
            }
        )
        dim_map = {
            "time": Dimension.TIME,
            "z": Dimension.Z_CENTER,
            "y": Dimension.Y_CENTER,
            "xi_u": Dimension.X_INNER,
        }
        fs = Fieldset.from_xarray(ds, dim_map)
        assert fs.x_size == NX

    def test_raises_if_size_cannot_be_inferred(self) -> None:
        # Dataset without any z dimension – z_size cannot be inferred.
        ds = xr.Dataset(
            {"h": xr.DataArray(np.ones((NY, NX)), dims=["y", "x"])}
        )
        dim_map = {
            "time": Dimension.TIME,  # time not in ds either
            "z": Dimension.Z_CENTER,
            "y": Dimension.Y_CENTER,
            "x": Dimension.X_CENTER,
        }
        with pytest.raises(ValueError, match="t_size"):
            Fieldset.from_xarray(ds, dim_map)

    def test_explicit_size_resolves_missing_inferred_size(self) -> None:
        ds = xr.Dataset(
            {"h": xr.DataArray(np.ones((NY, NX)), dims=["y", "x"])}
        )
        dim_map = {
            "y": Dimension.Y_CENTER,
            "x": Dimension.X_CENTER,
        }
        fs = Fieldset.from_xarray(ds, dim_map, t_size=T, z_size=NZ)
        assert fs.t_size == T
        assert fs.z_size == NZ


# ---------------------------------------------------------------------------
# Fields created correctly
# ---------------------------------------------------------------------------


class TestFromXarrayFieldCreation:
    def test_all_variables_become_fields(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, DIM_MAP)
        assert "u" in fs
        assert "h" in fs
        assert "C" in fs

    def test_time_dep_variable_is_time_dep_field(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, DIM_MAP)
        assert isinstance(fs["u"], TimeDependentField)

    def test_static_variable_is_static_field(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, DIM_MAP)
        assert isinstance(fs["h"], StaticField)

    def test_1d_variable_has_correct_invariant_staggers(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, DIM_MAP)
        C = fs["C"]
        assert isinstance(C, StaticField)
        assert C.z_stagger is Stagger.CENTER
        assert C.y_stagger is Stagger.INVARIANT
        assert C.x_stagger is Stagger.INVARIANT

    def test_2d_static_has_correct_staggers(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, DIM_MAP)
        h = fs["h"]
        assert h.z_stagger is Stagger.INVARIANT
        assert h.y_stagger is Stagger.CENTER
        assert h.x_stagger is Stagger.CENTER

    def test_full_3d_time_dep_field_has_correct_staggers(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, DIM_MAP)
        u = fs["u"]
        assert u.z_stagger is Stagger.CENTER
        assert u.y_stagger is Stagger.CENTER
        assert u.x_stagger is Stagger.CENTER

    def test_dask_backed_dataset_creates_fields(self) -> None:
        ds = xr.Dataset(
            {
                "u": xr.DataArray(
                    da.ones((T, NZ, NY, NX), chunks=(1, NZ, NY, NX)),
                    dims=["time", "z", "y", "x"],
                ),
            }
        )
        fs = Fieldset.from_xarray(ds, DIM_MAP)
        assert isinstance(fs["u"], TimeDependentField)


# ---------------------------------------------------------------------------
# Constants and index bounds
# ---------------------------------------------------------------------------


class TestFromXarrayConstantsAndBounds:
    def test_extra_constants_are_added(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, DIM_MAP, constants={"gravity": 9.81})
        assert "gravity" in fs.constants
        assert float(fs.constants["gravity"]) == pytest.approx(9.81)

    def test_custom_index_bounds_are_set(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(
            ds,
            DIM_MAP,
            zidx_bounds=(1, NZ - 2),
            yidx_bounds=(0.5, NY - 1.5),
            xidx_bounds=(-0.5, NX - 0.5),
        )
        assert fs.zidx_bounds == (np.float64(1), np.float64(NZ - 2))

    def test_simulation_shape_is_correct(self) -> None:
        ds = _make_dataset()
        fs = Fieldset.from_xarray(ds, DIM_MAP)
        assert fs.simulation_shape == (T, NZ, NY, NX)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestFromXarrayErrors:
    def test_raises_type_error_for_non_dataset(self) -> None:
        with pytest.raises(TypeError, match="Dataset"):
            Fieldset.from_xarray(np.ones((3, 4)), DIM_MAP)  # type: ignore[arg-type]

    def test_raises_for_inconsistent_centered_sizes(self) -> None:
        # Two z dims that imply different centred sizes
        ds = xr.Dataset(
            {
                "u": xr.DataArray(np.ones((NZ, NY, NX)), dims=["z_rho", "y", "x"]),
                "v": xr.DataArray(np.ones((NZ + 5, NY, NX)), dims=["z_wrong", "y", "x"]),
            }
        )
        dim_map = {
            "z_rho": Dimension.Z_CENTER,
            "z_wrong": Dimension.Z_CENTER,  # same direction, different size
            "y": Dimension.Y_CENTER,
            "x": Dimension.X_CENTER,
        }
        with pytest.raises(ValueError, match="Inconsistent"):
            Fieldset.from_xarray(ds, dim_map)
