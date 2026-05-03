"""Tests for the Output class."""

import numpy as np
import pytest

from offline_particles.output import Output


class TestOutputConstruction:
    def test_default_dtype_is_float64(self) -> None:
        output = Output("xidx")
        assert output.particle_property.dtype == np.dtype(np.float64)

    def test_explicit_dtype(self) -> None:
        output = Output("xidx", dtype=np.float32)
        assert output.particle_property.dtype == np.dtype(np.float32)

    def test_particle_property_name(self) -> None:
        output = Output("my_property")
        assert output.particle_property.name == "my_property"

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


class TestOutputRequiredPropertyDtypes:
    def test_includes_particle_property(self) -> None:
        output = Output("xidx", dtype=np.float32)
        required = output.required_property_dtypes
        assert "xidx" in required
        assert required["xidx"] == np.dtype(np.float32)

    def test_is_read_only_mapping(self) -> None:
        output = Output("xidx")
        required = output.required_property_dtypes
        with pytest.raises(TypeError):
            required["new_key"] = np.dtype(np.float64)  # type: ignore[index]
