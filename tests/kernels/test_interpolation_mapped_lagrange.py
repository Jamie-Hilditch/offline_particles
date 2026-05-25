"""Tests for mapped Lagrange interpolation kernel factories.

These tests focus on the wrapper logic that maps particle indices from the
canonical ``(zidx, yidx, xidx)`` order to the order required by the field
layout. The underlying interpolation math is already exercised in the base
Lagrange test module, so these tests validate the mapping contract directly.
"""

from itertools import permutations

import numba
import numpy as np
import numpy.typing as npt
import pytest

import offline_particles.kernels.interpolation._lagrange_mapped as lagrange_mapped_module
from offline_particles.kernels.interpolation import lagrange2N_mapped_particle_factory
from offline_particles.spatial_arrays import ArrayAxis

_PARTICLE_COORDS = {
    ArrayAxis.Z: 1.25,
    ArrayAxis.Y: 2.5,
    ArrayAxis.X: 3.75,
}

_OFFSETS = {
    ArrayAxis.Z: 0.125,
    ArrayAxis.Y: -0.25,
    ArrayAxis.X: 0.5,
}

_SPY_MAX_IDXS = {
    ArrayAxis.Z: 11,
    ArrayAxis.Y: 22,
    ArrayAxis.X: 33,
}

_CANONICAL_PARTICLE_COORDS = (
    _PARTICLE_COORDS[ArrayAxis.Z],
    _PARTICLE_COORDS[ArrayAxis.Y],
    _PARTICLE_COORDS[ArrayAxis.X],
)


def _particle_coords_in_layout(layout: tuple[ArrayAxis, ...]) -> tuple[float, ...]:
    return tuple(_PARTICLE_COORDS[axis] for axis in layout)


def _offsets_in_layout(layout: tuple[ArrayAxis, ...]) -> tuple[float, ...]:
    return tuple(_OFFSETS[axis] for axis in layout)


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
        coefficient * (_PARTICLE_COORDS[axis] + offset)
        for coefficient, axis, offset in zip((1.0, 10.0, 100.0)[: len(layout)], layout, offsets, strict=True)
    )


def _max_idxs_for_shape(shape: tuple[int, ...], n: int) -> tuple[int, ...]:
    return tuple(size - 2 * n for size in shape)


@pytest.mark.parametrize(
    "layout,n,expected_arity",
    [
        ((ArrayAxis.Z,), 2, 1),
        ((ArrayAxis.Y,), 2, 1),
        ((ArrayAxis.X,), 2, 1),
        ((ArrayAxis.Z, ArrayAxis.Y), 2, 2),
        ((ArrayAxis.Z, ArrayAxis.X), 2, 2),
        ((ArrayAxis.Y, ArrayAxis.Z), 2, 2),
        ((ArrayAxis.Y, ArrayAxis.X), 2, 2),
        ((ArrayAxis.X, ArrayAxis.Z), 2, 2),
        ((ArrayAxis.X, ArrayAxis.Y), 2, 2),
        ((ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X), 2, 3),
        ((ArrayAxis.Z, ArrayAxis.X, ArrayAxis.Y), 2, 3),
        ((ArrayAxis.Y, ArrayAxis.Z, ArrayAxis.X), 2, 3),
        ((ArrayAxis.Y, ArrayAxis.X, ArrayAxis.Z), 2, 3),
        ((ArrayAxis.X, ArrayAxis.Z, ArrayAxis.Y), 2, 3),
        ((ArrayAxis.X, ArrayAxis.Y, ArrayAxis.Z), 2, 3),
    ],
)
def test_mapped_factory_forwards_particle_coordinates_and_offsets(
    monkeypatch: pytest.MonkeyPatch,
    layout: tuple[ArrayAxis, ...],
    n: int,
    expected_arity: int,
) -> None:
    """The wrapper should pass coordinates in field-layout order to the particle factory."""
    called: dict[str, int] = {}

    if expected_arity == 1:

        @numba.njit(nogil=True, fastmath=True)
        def fake_interpolator(
            field_array: np.ndarray,
            offset_idx: float,
            max_idx: int,
        ) -> float:
            return offset_idx * 1000.0 + float(max_idx)

        def fake_factory(received_n: int):
            called["N"] = received_n
            return fake_interpolator

        monkeypatch.setattr(lagrange_mapped_module, "lagrange2N_1D_particle_factory", fake_factory)

        impl = lagrange2N_mapped_particle_factory(layout, n)
        field = np.ones((8,), dtype=np.float64)
        result = impl(
            _CANONICAL_PARTICLE_COORDS[0],
            _CANONICAL_PARTICLE_COORDS[1],
            _CANONICAL_PARTICLE_COORDS[2],
            field,
            _OFFSETS[layout[0]],
            _SPY_MAX_IDXS[layout[0]],
        )

        expected = (_PARTICLE_COORDS[layout[0]] + _OFFSETS[layout[0]]) * 1000.0 + float(_SPY_MAX_IDXS[layout[0]])

    elif expected_arity == 2:

        @numba.njit(nogil=True, fastmath=True)
        def fake_interpolator(
            field_array: np.ndarray,
            offset_idx_0: float,
            offset_idx_1: float,
            max_idx_0: int,
            max_idx_1: int,
        ) -> float:
            return offset_idx_0 * 1000.0 + offset_idx_1 * 100.0 + float(max_idx_0) * 10.0 + float(max_idx_1)

        def fake_factory(received_n: int):
            called["N"] = received_n
            return fake_interpolator

        monkeypatch.setattr(lagrange_mapped_module, "lagrange2N_2D_particle_factory", fake_factory)

        impl = lagrange2N_mapped_particle_factory(layout, n)
        field = np.ones((8, 9), dtype=np.float64)
        offsets = _offsets_in_layout(layout)
        max_idxs = tuple(_SPY_MAX_IDXS[axis] for axis in layout)
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

        coords = _particle_coords_in_layout(layout)
        expected = (coords[0] + offsets[0]) * 1000.0
        expected += (coords[1] + offsets[1]) * 100.0
        expected += float(max_idxs[0]) * 10.0 + float(max_idxs[1])

    else:

        @numba.njit(nogil=True, fastmath=True)
        def fake_interpolator(
            field_array: np.ndarray,
            offset_idx_0: float,
            offset_idx_1: float,
            offset_idx_2: float,
            max_idx_0: int,
            max_idx_1: int,
            max_idx_2: int,
        ) -> float:
            return (
                offset_idx_0 * 1000.0
                + offset_idx_1 * 100.0
                + offset_idx_2 * 10.0
                + float(max_idx_0)
                + float(max_idx_1) / 10.0
                + float(max_idx_2) / 100.0
            )

        def fake_factory(received_n: int):
            called["N"] = received_n
            return fake_interpolator

        monkeypatch.setattr(lagrange_mapped_module, "lagrange2N_3D_particle_factory", fake_factory)

        impl = lagrange2N_mapped_particle_factory(layout, n)
        field = np.ones((8, 9, 10), dtype=np.float64)
        offsets = _offsets_in_layout(layout)
        max_idxs = tuple(_SPY_MAX_IDXS[axis] for axis in layout)
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

        coords = _particle_coords_in_layout(layout)
        expected = (coords[0] + offsets[0]) * 1000.0
        expected += (coords[1] + offsets[1]) * 100.0
        expected += (coords[2] + offsets[2]) * 10.0
        expected += float(max_idxs[0]) + float(max_idxs[1]) / 10.0 + float(max_idxs[2]) / 100.0

    assert called == {"N": n}
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("layout", list(permutations((ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X), 1)))
def test_mapped_1d_interpolation_matches_affine_field(layout: tuple[ArrayAxis, ...]) -> None:
    """1D mapped interpolation should match a field that is linear in the mapped axis."""
    n = 2
    shape = (10,)
    field = 7.0 + 1.0 * np.arange(shape[0], dtype=np.float64)
    impl = lagrange2N_mapped_particle_factory(layout, n)

    offsets = _offsets_in_layout(layout)
    max_idxs = _max_idxs_for_shape(shape, n)
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
    n = 2
    shape = (10, 11)
    field = _build_affine_field(shape)
    impl = lagrange2N_mapped_particle_factory(layout, n)

    offsets = _offsets_in_layout(layout)
    max_idxs = _max_idxs_for_shape(shape, n)
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
    n = 2
    shape = (10, 11, 12)
    field = _build_affine_field(shape)
    impl = lagrange2N_mapped_particle_factory(layout, n)

    offsets = _offsets_in_layout(layout)
    max_idxs = _max_idxs_for_shape(shape, n)
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


@pytest.mark.parametrize("invalid_layout", [()])
def test_mapped_factory_rejects_unsupported_dimension_counts(
    invalid_layout: tuple[ArrayAxis, ...],
) -> None:
    with pytest.raises(ValueError, match="Unsupported number of dimensions"):
        lagrange2N_mapped_particle_factory(invalid_layout, 2)


@pytest.mark.parametrize("invalid_n", [0, -1])
def test_mapped_factory_propagates_invalid_N(invalid_n: int) -> None:
    with pytest.raises(ValueError, match="N must be a positive integer"):
        lagrange2N_mapped_particle_factory((ArrayAxis.Z,), invalid_n)


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
        lagrange2N_mapped_particle_factory(dim_ordering, 2)
