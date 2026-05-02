from offline_particles.spatial_arrays import (
    ALL_STAGGERS,
    CENTERED_STAGGERS,
    ON_FACE_STAGGERS,
    Stagger,
)


def test_stagger_location_is_exclusive() -> None:
    # Every stagger must be classified as exactly one of: on-face, or at-center.
    for stagger in Stagger:
        flags = (stagger.on_face, stagger.at_center)
        assert sum(flags) == 1


def test_location_stagger_sets_union_to_all_staggers() -> None:
    # Centered and on-face categories should cover all stagger values.
    union = CENTERED_STAGGERS | ON_FACE_STAGGERS
    assert union == ALL_STAGGERS


def test_location_stagger_sets_are_pairwise_non_intersecting() -> None:
    # Category sets should be disjoint from each other.
    assert CENTERED_STAGGERS.isdisjoint(ON_FACE_STAGGERS)
