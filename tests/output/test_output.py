"""Tests for the Output class."""

import numpy as np
import pytest

from offline_particles.output import Output


class TestOutputConstruction:
    def test_default_dtype_is_none(self) -> None:
        output = Output("xidx")
        assert output.dtype is None

    def test_explicit_dtype(self) -> None:
        output = Output("xidx", dtype=np.float32)
        assert output.dtype == np.dtype(np.float32)

    def test_particle_property_name(self) -> None:
        output = Output("my_property")
        assert output.particle_property == "my_property"

    def test_no_kernels_by_default(self) -> None:
        output = Output("xidx")
        assert output.kernels == ()

    def test_attrs_empty_by_default(self) -> None:
        output = Output("xidx")
        assert output.attrs == {}

    def test_attrs_set_via_kwargs(self) -> None:
        output = Output("xidx", units="m", long_name="x position")
        assert output.attrs == {"units": "m", "long_name": "x position"}

    def test_is_immutable(self) -> None:
        output = Output("xidx")
        with pytest.raises(AttributeError):
            output.particle_property = output.particle_property  # type: ignore[misc]
