"""Tests for TwoKeyDict."""

import pytest

from offline_particles.output._output import TwoKeyDict


class TestTwoKeyDictBasicOperations:
    def test_empty_initially(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        assert len(d) == 0
        assert list(d) == []

    def test_set_and_get(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["outer", "inner"] = 42
        assert d["outer", "inner"] == 42

    def test_set_multiple_inner_keys(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["outer", "a"] = 1
        d["outer", "b"] = 2
        assert d["outer", "a"] == 1
        assert d["outer", "b"] == 2

    def test_set_multiple_outer_keys(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["outer1", "inner"] = 10
        d["outer2", "inner"] = 20
        assert d["outer1", "inner"] == 10
        assert d["outer2", "inner"] == 20

    def test_overwrite_value(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["outer", "inner"] = 1
        d["outer", "inner"] = 99
        assert d["outer", "inner"] == 99

    def test_get_missing_outer_raises(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        with pytest.raises(KeyError):
            _ = d["missing", "inner"]

    def test_get_missing_inner_raises(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["outer", "a"] = 1
        with pytest.raises(KeyError):
            _ = d["outer", "missing"]

    def test_delete_item(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["outer", "inner"] = 42
        del d["outer", "inner"]
        assert ("outer", "inner") not in d

    def test_delete_removes_empty_outer_key(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["outer", "inner"] = 42
        del d["outer", "inner"]
        assert len(d.outer_keys()) == 0

    def test_delete_preserves_other_inner_keys(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["outer", "a"] = 1
        d["outer", "b"] = 2
        del d["outer", "a"]
        assert ("outer", "a") not in d
        assert d["outer", "b"] == 2

    def test_delete_missing_key_raises(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        with pytest.raises(KeyError):
            del d["missing", "inner"]


class TestTwoKeyDictLen:
    def test_len_empty(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        assert len(d) == 0

    def test_len_single(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["a", "b"] = 1
        assert len(d) == 1

    def test_len_multiple_inner(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["a", "x"] = 1
        d["a", "y"] = 2
        d["b", "x"] = 3
        assert len(d) == 3

    def test_len_after_delete(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["a", "x"] = 1
        d["a", "y"] = 2
        del d["a", "x"]
        assert len(d) == 1


class TestTwoKeyDictIteration:
    def test_iter_empty(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        assert list(d) == []

    def test_iter_yields_tuples(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["a", "x"] = 1
        keys = list(d)
        assert keys == [("a", "x")]

    def test_iter_all_keys(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["a", "x"] = 1
        d["a", "y"] = 2
        d["b", "x"] = 3
        keys = set(d)
        assert keys == {("a", "x"), ("a", "y"), ("b", "x")}

    def test_contains(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["a", "x"] = 1
        assert ("a", "x") in d
        assert ("a", "y") not in d
        assert ("b", "x") not in d

    def test_items_iteration(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["a", "x"] = 1
        d["b", "y"] = 2
        items = dict(d.items())
        assert items == {("a", "x"): 1, ("b", "y"): 2}


class TestTwoKeyDictGetInnerMapping:
    def test_get_inner_mapping(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["outer", "a"] = 1
        d["outer", "b"] = 2
        inner = d.get_inner_mapping("outer")
        assert dict(inner) == {"a": 1, "b": 2}

    def test_get_inner_mapping_missing_key_raises(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        with pytest.raises(KeyError, match="missing"):
            d.get_inner_mapping("missing")

    def test_get_inner_mapping_is_read_only(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["outer", "a"] = 1
        inner = d.get_inner_mapping("outer")
        with pytest.raises(TypeError):
            inner["a"] = 99  # type: ignore[index]


class TestTwoKeyDictGetOuterKeys:
    def test_get_outer_keys_empty(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        assert len(d.outer_keys()) == 0

    def test_get_outer_keys_single(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["outer", "inner"] = 1
        assert list(d.outer_keys()) == ["outer"]

    def test_get_outer_keys_multiple(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["a", "x"] = 1
        d["b", "y"] = 2
        d["a", "z"] = 3
        outer_keys = d.outer_keys()
        assert set(outer_keys) == {"a", "b"}

    def test_get_outer_keys_removed_after_delete(self) -> None:
        d: TwoKeyDict[str, str, int] = TwoKeyDict()
        d["a", "x"] = 1
        del d["a", "x"]
        assert "a" not in d.outer_keys()
