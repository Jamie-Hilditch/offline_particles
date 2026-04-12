import pytest

from offline_particles.spatial_arrays import ArrayAxis
def test_array_axis_has_exactly_three_canonical_members() -> None:
    members = list(ArrayAxis)

    assert len(members) == 3
    assert {member.name for member in members} == {"Z", "Y", "X"}
    assert {member.value for member in members} == {"Z", "Y", "X"}


def test_array_axis_aliases_resolve_to_canonical_members() -> None:
    assert ArrayAxis.DEPTH is ArrayAxis.Z
    assert ArrayAxis.VERTICAL is ArrayAxis.Z

    assert ArrayAxis.LATITUDE is ArrayAxis.Y
    assert ArrayAxis.LAT is ArrayAxis.Y
    assert ArrayAxis.MERIDIONAL is ArrayAxis.Y

    assert ArrayAxis.LONGITUDE is ArrayAxis.X
    assert ArrayAxis.LON is ArrayAxis.X
    assert ArrayAxis.ZONAL is ArrayAxis.X


def test_array_axis_members_mapping_includes_all_aliases() -> None:
    expected = {
        "Z": ArrayAxis.Z,
        "DEPTH": ArrayAxis.Z,
        "VERTICAL": ArrayAxis.Z,
        "Y": ArrayAxis.Y,
        "LATITUDE": ArrayAxis.Y,
        "LAT": ArrayAxis.Y,
        "MERIDIONAL": ArrayAxis.Y,
        "X": ArrayAxis.X,
        "LONGITUDE": ArrayAxis.X,
        "LON": ArrayAxis.X,
        "ZONAL": ArrayAxis.X,
    }

    assert ArrayAxis.__members__ == expected


def test_array_axis_string_value_constructor_uses_canonical_values() -> None:
    assert ArrayAxis("Z") is ArrayAxis.Z
    assert ArrayAxis("Y") is ArrayAxis.Y
    assert ArrayAxis("X") is ArrayAxis.X


class TestArrayAxisParse:
    def test_parse_canonical_values(self) -> None:
        assert ArrayAxis.parse("Z") is ArrayAxis.Z
        assert ArrayAxis.parse("Y") is ArrayAxis.Y
        assert ArrayAxis.parse("X") is ArrayAxis.X

    def test_parse_alias_names(self) -> None:
        assert ArrayAxis.parse("DEPTH") is ArrayAxis.Z
        assert ArrayAxis.parse("VERTICAL") is ArrayAxis.Z
        assert ArrayAxis.parse("LATITUDE") is ArrayAxis.Y
        assert ArrayAxis.parse("LAT") is ArrayAxis.Y
        assert ArrayAxis.parse("MERIDIONAL") is ArrayAxis.Y
        assert ArrayAxis.parse("LONGITUDE") is ArrayAxis.X
        assert ArrayAxis.parse("LON") is ArrayAxis.X
        assert ArrayAxis.parse("ZONAL") is ArrayAxis.X

    def test_parse_enum_member_passthrough(self) -> None:
        assert ArrayAxis.parse(ArrayAxis.Z) is ArrayAxis.Z
        assert ArrayAxis.parse(ArrayAxis.Y) is ArrayAxis.Y
        assert ArrayAxis.parse(ArrayAxis.X) is ArrayAxis.X

    def test_parse_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid ArrayAxis"):
            ArrayAxis.parse("INVALID")

    def test_parse_lowercase_raises(self) -> None:
        """Lowercase strings are not valid — values and names are uppercase."""
        with pytest.raises(ValueError, match="not a valid ArrayAxis"):
            ArrayAxis.parse("z")
