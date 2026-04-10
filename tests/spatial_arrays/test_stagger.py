from offline_particles.spatial_arrays import (
    ACTIVE_STAGGERS,
    ALL_STAGGERS,
    CENTERED_STAGGERS,
    INACTIVE_STAGGERS,
    INVARIANT_STAGGERS,
    ON_FACE_STAGGERS,
    Stagger,
)


def test_stagger_location_is_exclusive() -> None:
    # Every stagger must be classified as exactly one of: invariant, on-face, or at-center.
    for stagger in Stagger:
        flags = (stagger.is_invariant, stagger.on_face, stagger.at_center)
        assert sum(flags) == 1


def test_stagger_is_active_and_is_invariant_are_exclusive() -> None:
    # Exactly one of is_active or is_invariant must be true.
    for stagger in Stagger:
        assert stagger.is_active != stagger.is_invariant


def test_stagger_offset_is_float_if_active() -> None:
    # Active staggers should always provide a float offset.
    for stagger in Stagger:
        if stagger.is_active:
            assert isinstance(stagger.offset, float)


def test_stagger_offset_is_none_if_inactive() -> None:
    # Inactive (invariant) staggers have no offset.
    for stagger in Stagger:
        if not stagger.is_active:
            assert stagger.offset is None


def test_stagger_expected_size_is_int_if_active() -> None:
    # Active staggers should produce an integer expected size.
    for stagger in Stagger:
        if stagger.is_active:
            assert isinstance(stagger.expected_size(7), int)


def test_stagger_expected_size_is_none_if_invariant() -> None:
    # Invariant staggers should not define an expected size.
    for stagger in Stagger:
        if stagger.is_invariant:
            assert stagger.expected_size(7) is None


def test_location_stagger_sets_union_to_all_staggers() -> None:
    # Centered, on-face, and invariant categories should cover all stagger values.
    union = CENTERED_STAGGERS | ON_FACE_STAGGERS | INVARIANT_STAGGERS
    assert union == ALL_STAGGERS


def test_location_stagger_sets_are_pairwise_non_intersecting() -> None:
    # Category sets should be disjoint from each other.
    assert CENTERED_STAGGERS.isdisjoint(ON_FACE_STAGGERS)
    assert CENTERED_STAGGERS.isdisjoint(INVARIANT_STAGGERS)
    assert ON_FACE_STAGGERS.isdisjoint(INVARIANT_STAGGERS)


def test_active_and_inactive_stagger_sets_union_to_all_staggers() -> None:
    # Active and inactive categories should cover all staggers.
    union = ACTIVE_STAGGERS | INACTIVE_STAGGERS
    assert union == ALL_STAGGERS


def test_active_and_inactive_stagger_sets_are_disjoint() -> None:
    # Active and inactive categories should not overlap.
    assert ACTIVE_STAGGERS.isdisjoint(INACTIVE_STAGGERS)
