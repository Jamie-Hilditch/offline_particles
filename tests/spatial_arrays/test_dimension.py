"""Tests for the Dimension enum."""

import pytest

from offline_particles.spatial_arrays import Dimension, Stagger

# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestDimensionBasicStructure:
    def test_time_member_exists(self) -> None:
        assert hasattr(Dimension, "TIME")

    def test_time_direction(self) -> None:
        assert Dimension.TIME.direction == "T"

    def test_time_stagger_is_none(self) -> None:
        assert Dimension.TIME.stagger is None

    def test_time_is_time(self) -> None:
        assert Dimension.TIME.is_time is True

    def test_time_is_not_spatial(self) -> None:
        assert Dimension.TIME.is_spatial is False

    def test_canonical_z_members_exist(self) -> None:
        for stagger in Stagger:
            assert hasattr(Dimension, f"Z_{stagger.name}")

    def test_canonical_y_members_exist(self) -> None:
        for stagger in Stagger:
            assert hasattr(Dimension, f"Y_{stagger.name}")

    def test_canonical_x_members_exist(self) -> None:
        for stagger in Stagger:
            assert hasattr(Dimension, f"X_{stagger.name}")

    def test_unique_member_count(self) -> None:
        # 1 TIME + 3 directions * len(Stagger) unique members
        assert len(list(Dimension)) == 1 + 3 * len(Stagger)


# ---------------------------------------------------------------------------
# Spatial member properties
# ---------------------------------------------------------------------------


class TestDimensionSpatialProperties:
    @pytest.mark.parametrize("stagger", list(Stagger))
    def test_z_direction(self, stagger: Stagger) -> None:
        assert Dimension[f"Z_{stagger.name}"].direction == "Z"

    @pytest.mark.parametrize("stagger", list(Stagger))
    def test_y_direction(self, stagger: Stagger) -> None:
        assert Dimension[f"Y_{stagger.name}"].direction == "Y"

    @pytest.mark.parametrize("stagger", list(Stagger))
    def test_x_direction(self, stagger: Stagger) -> None:
        assert Dimension[f"X_{stagger.name}"].direction == "X"

    @pytest.mark.parametrize("stagger", list(Stagger))
    def test_z_stagger_value(self, stagger: Stagger) -> None:
        assert Dimension[f"Z_{stagger.name}"].stagger is stagger

    @pytest.mark.parametrize("stagger", list(Stagger))
    def test_spatial_is_not_time(self, stagger: Stagger) -> None:
        assert Dimension[f"Z_{stagger.name}"].is_time is False
        assert Dimension[f"Y_{stagger.name}"].is_time is False
        assert Dimension[f"X_{stagger.name}"].is_time is False

    @pytest.mark.parametrize("stagger", list(Stagger))
    def test_spatial_is_spatial(self, stagger: Stagger) -> None:
        assert Dimension[f"Z_{stagger.name}"].is_spatial is True
        assert Dimension[f"Y_{stagger.name}"].is_spatial is True
        assert Dimension[f"X_{stagger.name}"].is_spatial is True


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


class TestDimensionAliases:
    @pytest.mark.parametrize("stagger", list(Stagger))
    def test_depth_is_alias_for_z(self, stagger: Stagger) -> None:
        assert Dimension[f"DEPTH_{stagger.name}"] is Dimension[f"Z_{stagger.name}"]

    @pytest.mark.parametrize("stagger", list(Stagger))
    def test_eta_is_alias_for_y(self, stagger: Stagger) -> None:
        assert Dimension[f"ETA_{stagger.name}"] is Dimension[f"Y_{stagger.name}"]

    @pytest.mark.parametrize("stagger", list(Stagger))
    def test_xi_is_alias_for_x(self, stagger: Stagger) -> None:
        assert Dimension[f"XI_{stagger.name}"] is Dimension[f"X_{stagger.name}"]

    def test_depth_inner_is_z_inner(self) -> None:
        assert Dimension.DEPTH_INNER is Dimension.Z_INNER

    def test_eta_center_is_y_center(self) -> None:
        assert Dimension.ETA_CENTER is Dimension.Y_CENTER

    def test_xi_outer_is_x_outer(self) -> None:
        assert Dimension.XI_OUTER is Dimension.X_OUTER
