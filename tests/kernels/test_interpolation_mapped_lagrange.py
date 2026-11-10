"""Tests for mapped Lagrange interpolation kernel factories.

These tests focus on the wrapper logic that maps particle indices from the
canonical ``(zidx, yidx, xidx)`` order to the order required by the field
layout, and on `lagrange_mapped_particle_factory`'s own per-axis `N`/`D` tuple
contract. The underlying interpolation math is already exercised in the base
Lagrange test module, so these tests validate the mapping and parameter
threading, not the polynomial math itself.
"""

from itertools import permutations

import numpy as np
import numpy.typing as npt
import pytest

from offline_particles.kernels.interpolation import lagrange_mapped_particle_factory
from offline_particles.spatial_arrays import ArrayAxis
from tests.kernels.conftest import PARTICLE_COORDS, offsets_in_layout

_SPY_MAX_IDXS = {
    ArrayAxis.Z: 11,
    ArrayAxis.Y: 22,
    ArrayAxis.X: 33,
}

_CANONICAL_PARTICLE_COORDS = (
    PARTICLE_COORDS[ArrayAxis.Z],
    PARTICLE_COORDS[ArrayAxis.Y],
    PARTICLE_COORDS[ArrayAxis.X],
)


def _build_affine_field(shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """Build a field whose values are an affine function of each array axis.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the field to build.

    Returns
    -------
    npt.NDArray[np.float64]
        The affine field.
    """
    grids = np.meshgrid(*(np.arange(size, dtype=np.float64) for size in shape), indexing="ij")
    field = np.full(shape, 7.0, dtype=np.float64)
    for coefficient, grid in zip((1.0, 10.0, 100.0)[: len(shape)], grids, strict=True):
        field = field + coefficient * grid
    return field


def _expected_affine_value(layout: tuple[ArrayAxis, ...], offsets: tuple[float, ...]) -> float:
    return 7.0 + sum(
        coefficient * (PARTICLE_COORDS[axis] + offset)
        for coefficient, axis, offset in zip((1.0, 10.0, 100.0)[: len(layout)], layout, offsets, strict=True)
    )


def _max_idxs_for_shape(shape: tuple[int, ...], ns: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(size - 2 * n for size, n in zip(shape, ns, strict=True))


@pytest.mark.parametrize("layout", list(permutations((ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X), 1)))
def test_mapped_1d_interpolation_matches_affine_field(layout: tuple[ArrayAxis, ...]) -> None:
    """1D mapped interpolation should match a field that is linear in the mapped axis."""
    ns = (2,)
    shape = (10,)
    field = 7.0 + 1.0 * np.arange(shape[0], dtype=np.float64)
    impl = lagrange_mapped_particle_factory(layout, ns)

    offsets = offsets_in_layout(layout)
    max_idxs = _max_idxs_for_shape(shape, ns)
    result = impl(
        _CANONICAL_PARTICLE_COORDS[0],
        _CANONICAL_PARTICLE_COORDS[1],
        _CANONICAL_PARTICLE_COORDS[2],
        field,
        offsets[0],
        max_idxs[0],
    )

    assert result == pytest.approx(_expected_affine_value(layout, offsets))


@pytest.mark.parametrize("layout", list(permutations((ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X), 2)))
def test_mapped_2d_interpolation_matches_affine_field(layout: tuple[ArrayAxis, ...]) -> None:
    """2D mapped interpolation should match a field that is linear in both mapped axes."""
    ns = (2, 2)
    shape = (10, 11)
    field = _build_affine_field(shape)
    impl = lagrange_mapped_particle_factory(layout, ns)

    offsets = offsets_in_layout(layout)
    max_idxs = _max_idxs_for_shape(shape, ns)
    result = impl(
        _CANONICAL_PARTICLE_COORDS[0],
        _CANONICAL_PARTICLE_COORDS[1],
        _CANONICAL_PARTICLE_COORDS[2],
        field,
        offsets[0],
        offsets[1],
        max_idxs[0],
        max_idxs[1],
    )

    assert result == pytest.approx(_expected_affine_value(layout, offsets))


@pytest.mark.parametrize("layout", list(permutations((ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X), 2)))
def test_mapped_2d_interpolation_accepts_different_N_per_axis(layout: tuple[ArrayAxis, ...]) -> None:
    """Distinct per-axis N should still reproduce an affine field exactly (affine is exact for any N >= 1)."""
    ns = (1, 3)
    shape = (10, 11)
    field = _build_affine_field(shape)
    impl = lagrange_mapped_particle_factory(layout, ns)

    offsets = offsets_in_layout(layout)
    max_idxs = _max_idxs_for_shape(shape, ns)
    result = impl(
        _CANONICAL_PARTICLE_COORDS[0],
        _CANONICAL_PARTICLE_COORDS[1],
        _CANONICAL_PARTICLE_COORDS[2],
        field,
        offsets[0],
        offsets[1],
        max_idxs[0],
        max_idxs[1],
    )

    assert result == pytest.approx(_expected_affine_value(layout, offsets))


@pytest.mark.parametrize("layout", list(permutations((ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X), 3)))
def test_mapped_3d_interpolation_matches_affine_field(layout: tuple[ArrayAxis, ...]) -> None:
    """3D mapped interpolation should match a field that is linear in all mapped axes."""
    ns = (2, 2, 2)
    shape = (10, 11, 12)
    field = _build_affine_field(shape)
    impl = lagrange_mapped_particle_factory(layout, ns)

    offsets = offsets_in_layout(layout)
    max_idxs = _max_idxs_for_shape(shape, ns)
    result = impl(
        _CANONICAL_PARTICLE_COORDS[0],
        _CANONICAL_PARTICLE_COORDS[1],
        _CANONICAL_PARTICLE_COORDS[2],
        field,
        offsets[0],
        offsets[1],
        offsets[2],
        max_idxs[0],
        max_idxs[1],
        max_idxs[2],
    )

    assert result == pytest.approx(_expected_affine_value(layout, offsets))


def test_mapped_1d_interpolation_derivative_order_is_threaded_through() -> None:
    """D should reach the underlying particle factory even through the index-mapping layer."""
    layout = (ArrayAxis.Y,)
    shape = (10,)
    # f(x) = 3 + 2x; df/dx == 2 everywhere, independent of position
    field = 3.0 + 2.0 * np.arange(shape[0], dtype=np.float64)
    impl = lagrange_mapped_particle_factory(layout, (2,), (1,))

    offsets = offsets_in_layout(layout)
    max_idxs = _max_idxs_for_shape(shape, (2,))
    result = impl(
        _CANONICAL_PARTICLE_COORDS[0],
        _CANONICAL_PARTICLE_COORDS[1],
        _CANONICAL_PARTICLE_COORDS[2],
        field,
        offsets[0],
        max_idxs[0],
    )

    assert result == pytest.approx(2.0)


def test_mapped_factory_D_defaults_to_zero_for_every_axis() -> None:
    layout = (ArrayAxis.Z, ArrayAxis.Y)
    shape = (10, 11)
    field = _build_affine_field(shape)
    ns = (2, 2)

    impl_without_d = lagrange_mapped_particle_factory(layout, ns)
    impl_with_explicit_zero_d = lagrange_mapped_particle_factory(layout, ns, (0, 0))

    offsets = offsets_in_layout(layout)
    max_idxs = _max_idxs_for_shape(shape, ns)
    args = (
        _CANONICAL_PARTICLE_COORDS[0],
        _CANONICAL_PARTICLE_COORDS[1],
        _CANONICAL_PARTICLE_COORDS[2],
        field,
        offsets[0],
        offsets[1],
        max_idxs[0],
        max_idxs[1],
    )

    assert impl_without_d(*args) == pytest.approx(impl_with_explicit_zero_d(*args))


@pytest.mark.parametrize("invalid_layout", [()])
def test_mapped_factory_rejects_unsupported_dimension_counts(
    invalid_layout: tuple[ArrayAxis, ...],
) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        lagrange_mapped_particle_factory(invalid_layout, ())


def test_mapped_factory_rejects_mismatched_N_length() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        lagrange_mapped_particle_factory((ArrayAxis.Z, ArrayAxis.Y), (2,))


@pytest.mark.parametrize("invalid_n", [0, -1])
def test_mapped_factory_propagates_invalid_N(invalid_n: int) -> None:
    with pytest.raises(ValueError, match="N must be a positive integer"):
        lagrange_mapped_particle_factory((ArrayAxis.Z,), (invalid_n,))


@pytest.mark.parametrize("invalid_d", [-1, -5])
def test_mapped_factory_propagates_invalid_derivative_order(invalid_d: int) -> None:
    with pytest.raises(ValueError, match="derivative_order"):
        lagrange_mapped_particle_factory((ArrayAxis.Z,), (2,), (invalid_d,))


@pytest.mark.parametrize(
    "dim_ordering",
    [
        (ArrayAxis.X, ArrayAxis.X),
        (ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.Y),
        (ArrayAxis.X, ArrayAxis.Z, ArrayAxis.X),
    ],
)
def test_mapped_factory_rejects_duplicate_axes(dim_ordering: tuple[ArrayAxis, ...]) -> None:
    with pytest.raises(ValueError, match="Duplicate dimensions in dim_ordering"):
        lagrange_mapped_particle_factory(dim_ordering, tuple(2 for _ in dim_ordering))
