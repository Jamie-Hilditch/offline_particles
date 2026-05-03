"""Functions that validate array layouts."""

from ..spatial_arrays import ArrayAxis, ArrayLayout


# layout validators
def validate_ZYX_ordering(layout: ArrayLayout) -> None:
    """Validate that the layout has Z, Y, X axes in that order (with any staggers)."""
    expected_axes = (ArrayAxis.Z, ArrayAxis.Y, ArrayAxis.X)
    if layout.axes != expected_axes:
        raise ValueError(f"Expected axes {expected_axes} but got {layout.axes}")


def validate_ZY_ordering(layout: ArrayLayout) -> None:
    """Validate that the layout has Z, Y axes in that order (with any staggers)."""
    expected_axes = (ArrayAxis.Z, ArrayAxis.Y)
    if layout.axes != expected_axes:
        raise ValueError(f"Expected axes {expected_axes} but got {layout.axes}")


def validate_YX_ordering(layout: ArrayLayout) -> None:
    """Validate that the layout has Y, X axes in that order (with any staggers)."""
    expected_axes = (ArrayAxis.Y, ArrayAxis.X)
    if layout.axes != expected_axes:
        raise ValueError(f"Expected axes {expected_axes} but got {layout.axes}")


def validate_ZX_ordering(layout: ArrayLayout) -> None:
    """Validate that the layout has Z, X axes in that order (with any staggers)."""
    expected_axes = (ArrayAxis.Z, ArrayAxis.X)
    if layout.axes != expected_axes:
        raise ValueError(f"Expected axes {expected_axes} but got {layout.axes}")


def validate_Z_ordering(layout: ArrayLayout) -> None:
    """Validate that the layout has a single Z axis (with any staggers)."""
    expected_axes = (ArrayAxis.Z,)
    if layout.axes != expected_axes:
        raise ValueError(f"Expected axes {expected_axes} but got {layout.axes}")


def validate_Y_ordering(layout: ArrayLayout) -> None:
    """Validate that the layout has a single Y axis (with any staggers)."""
    expected_axes = (ArrayAxis.Y,)
    if layout.axes != expected_axes:
        raise ValueError(f"Expected axes {expected_axes} but got {layout.axes}")


def validate_X_ordering(layout: ArrayLayout) -> None:
    """Validate that the layout has a single X axis (with any staggers)."""
    expected_axes = (ArrayAxis.X,)
    if layout.axes != expected_axes:
        raise ValueError(f"Expected axes {expected_axes} but got {layout.axes}")
