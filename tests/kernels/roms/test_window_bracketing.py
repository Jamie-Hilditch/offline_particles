"""Regression tests for issue #147: a too-small `C` window makes `compute_zidx` silently wrong.

`_compute_Cidx_from_S`'s only documented contract on `C` is that it is strictly increasing; it has
no way to know whether the window it was handed actually covers the particle's true position. When
it doesn't, the binary search's boundary clamp (designed for genuine physical extrapolation beyond
the real water column) fires for the wrong reason and silently returns an index local to the wrong
window, with no error or signal. These tests pin down the correct invariant directly: windowing or
offsetting `C` must never change the computed `zidx`, regardless of how small the window is.

Uses a synthetic strictly-increasing, nonlinear `C` rather than the repo's existing `_LINEAR_C`
fixture (`test_vertical_coordinate_helpers.py`), since `C == sigma` there makes the S-coordinate
transform degenerate in a way that could mask an offset-arithmetic bug.
"""

import numpy as np
import pytest

from offline_particles.kernels.roms._vertical_coordinate import compute_zidx

_HC, _H, _ZETA, _NZ = 5.0, 45.0, 1.0, 20
# strictly increasing, nonlinear, spans [-1, 0]
_C_FULL = -1.0 + (np.arange(_NZ, dtype=np.float64) / (_NZ - 1)) ** 2
# true zidx ~= 10.42: comfortably in the interior, away from both array edges
_Z = -30.0


def _true_zidx() -> float:
    return compute_zidx(_Z, _H, _ZETA, _HC, _NZ, _C_FULL, 0.0)


@pytest.mark.xfail(
    reason="issue #147: window doesn't bracket the solution, compute_zidx silently returns the wrong zidx"
)
def test_compute_zidx_matches_full_array_even_when_window_is_too_short_to_bracket_the_solution() -> None:
    """Issue #147 minimal repro shape: same C_offset=0.0, just a shorter window."""
    assert _true_zidx() > 8  # sanity-check: the window below must not bracket this
    windowed = compute_zidx(_Z, _H, _ZETA, _HC, _NZ, _C_FULL[:8], 0.0)
    assert windowed == pytest.approx(_true_zidx())


@pytest.mark.xfail(
    reason="issue #147: window doesn't bracket the solution, compute_zidx silently returns the wrong zidx"
)
def test_compute_zidx_matches_full_array_even_when_a_nonzero_offset_window_is_too_short_to_bracket_the_solution() -> (
    None
):
    """Same defect, with a window shifted off index 0 - no existing test covers nonzero C_offset."""
    start = 2
    window = _C_FULL[start : start + 6]
    assert not (start <= _true_zidx() <= start + 5)  # sanity-check: window must not bracket this
    windowed = compute_zidx(_Z, _H, _ZETA, _HC, _NZ, window, -float(start))
    assert windowed == pytest.approx(_true_zidx())


def test_compute_zidx_matches_full_array_when_a_nonzero_offset_window_does_bracket_the_solution() -> None:
    """Control: a bracketing offset window still matches - isolates bracketing, not offset sign, as the cause."""
    start = 10
    window = _C_FULL[start:]
    assert start <= _true_zidx() <= start + len(window) - 2  # sanity-check: window must bracket this
    windowed = compute_zidx(_Z, _H, _ZETA, _HC, _NZ, window, -float(start))
    assert windowed == pytest.approx(_true_zidx())
