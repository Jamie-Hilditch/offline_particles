"""A module to test the convenience decorator constructors for KernelFunctions and ParticleKernels."""

from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import pytest

from offline_particles.fields import FieldData
from offline_particles.kernels import (
    FieldDataDeclaration,
    FieldDataType,
    ParticleKernel,
    ParticlePropertiesType,
    ParticlePropertyDeclaration,
    ScalarDeclaration,
    ScalarsType,
    kernel_function_impl,
    particle_kernel,
    particle_kernel_impl,
)


@pytest.fixture(scope="function")
def velocity_field_data() -> FieldData:
    """Create a sample velocity FieldData for testing.

    Returns
    -------
    FieldData
        A FieldData instance containing a 3D array and offsets.
    """
    array = np.zeros((10, 10, 10))
    offsets = (0.0, 0.0, 0.0)
    return FieldData(array=array, offsets=offsets)


@pytest.fixture(scope="function")
def metric_field_data() -> FieldData:
    """Create a sample metric FieldData for testing.

    Returns
    -------
    FieldData
        A FieldData instance containing a 2D array and offsets.
    """
    array = np.ones((10, 10))
    offsets = (0.0, 0.0)
    return FieldData(array=array, offsets=offsets)


@pytest.fixture(scope="function")
def index_particle_property_declaration() -> ParticlePropertyDeclaration:
    """Create a sample ParticlePropertyDeclaration for testing.

    Returns
    -------
    ParticlePropertyDeclaration
        A ParticlePropertyDeclaration instance for a particle index property.
    """
    return ParticlePropertyDeclaration("idx", np.float64, description="The index of the particle.")


@pytest.fixture(scope="function")
def index_tendency_particle_property_declaration() -> ParticlePropertyDeclaration:
    """Create a sample ParticlePropertyDeclaration for testing.

    Returns
    -------
    ParticlePropertyDeclaration
        A ParticlePropertyDeclaration instance for a particle index tendency property.
    """
    return ParticlePropertyDeclaration("didx_dt", np.float64, description="The tendency of the particle index.")


@pytest.fixture(scope="function")
def timestep_scalar_declaration() -> ScalarDeclaration:
    """Create a sample ScalarDeclaration for testing.

    Returns
    -------
    ScalarDeclaration
        A ScalarDeclaration instance for the timestep scalar.
    """
    return ScalarDeclaration("dt", np.float64, description="The timestep.")


@pytest.fixture(scope="function")
def velocity_field_data_declaration() -> FieldDataDeclaration:
    """Create a sample FieldDataDeclaration for testing.

    Returns
    -------
    FieldDataDeclaration
        A FieldDataDeclaration instance for a velocity field.
    """
    return FieldDataDeclaration("velocity", np.float32, [], "The velocity field data.")


@pytest.fixture(scope="function")
def metric_field_data_declaration() -> FieldDataDeclaration:
    """Create a sample FieldDataDeclaration for testing.

    Returns
    -------
    FieldDataDeclaration
        A FieldDataDeclaration instance for a metric field.
    """
    return FieldDataDeclaration("metric", np.float32, [], "The metric field data.")


@pytest.fixture(scope="function")
def particle_property_args() -> dict[str, npt.NDArray]:
    """Create sample particle property arguments for testing.

    Returns
    -------
    dict[str, npt.NDArray]
        A dictionary of particle property names to their corresponding arrays.
    """
    return {
        "idx": np.array([0.0, 1.0, 2.0]),
        "didx_dt": np.array([0.0, 0.0, 0.0]),
    }


@pytest.fixture(scope="function")
def scalar_args() -> dict[str, np.generic]:
    """Create sample scalar arguments for testing.

    Returns
    -------
    dict[str, np.generic]
        A dictionary of scalar names to their corresponding values.
    """
    return {
        "dt": np.float64(0.1),
    }


@pytest.fixture(scope="function")
def field_data_args(velocity_field_data: FieldData, metric_field_data: FieldData) -> dict[str, FieldData]:
    """Create sample field data arguments for testing.

    Parameters
    ----------
    velocity_field_data : FieldData
        A FieldData instance representing the velocity field.
    metric_field_data : FieldData
        A FieldData instance representing the metric field.

    Returns
    -------
    dict[str, FieldData]
        A dictionary of field data names to their corresponding FieldData instances.
    """
    return {
        "velocity": velocity_field_data,
        "metric": metric_field_data,
    }


def _particle_index_tendency_kernel_function_implementation(
    idx: npt.NDArray,
    didx_dt: npt.NDArray,
    dt: np.float64,
    velocity_array: npt.NDArray,
    velocity_offset_0: float,
    velocity_offset_1: float,
    velocity_offset_2: float,
    metric_array: npt.NDArray,
    metric_offset_0: float,
    metric_offset_1: float,
) -> None:
    """Perform a mock calculation."""
    # for testing purposes, just set the tendency to be the sum of the inputs
    didx_dt[:] = (
        idx
        + dt
        + velocity_array.sum()
        + velocity_offset_0
        + velocity_offset_1
        + velocity_offset_2
        + metric_array.sum()
        + metric_offset_0
        + metric_offset_1
    )


@pytest.fixture(scope="function")
def particle_index_tendency_kernel_function_implementation():
    return Mock(wraps=_particle_index_tendency_kernel_function_implementation)


@pytest.fixture(scope="function")
def particle_index_tendency_kernel_function(particle_index_tendency_kernel_function_implementation):
    def _kernel_function(
        particle_properties: ParticlePropertiesType,
        scalars: ScalarsType,
        field_data: FieldDataType,
    ) -> None:
        particle_index_tendency_kernel_function_implementation(
            particle_properties["idx"],
            particle_properties["didx_dt"],
            scalars["dt"],
            *field_data["velocity"].unpack(),
            *field_data["metric"].unpack(),
        )

    return Mock(wraps=_kernel_function)


@pytest.fixture(scope="function")
def particle_index_tendency_kernel(
    particle_index_tendency_kernel_function,
    index_particle_property_declaration,
    index_tendency_particle_property_declaration,
    timestep_scalar_declaration,
    velocity_field_data_declaration,
    metric_field_data_declaration,
) -> ParticleKernel:
    """Create a sample ParticleKernel for testing.

    Returns
    -------
    ParticleKernel
        A ParticleKernel instance representing a particle index tendency kernel.
    """
    return ParticleKernel(
        particle_index_tendency_kernel_function,
        particle_properties=[
            index_particle_property_declaration,
            index_tendency_particle_property_declaration,
        ],
        scalars=[timestep_scalar_declaration],
        field_data=[
            velocity_field_data_declaration,
            metric_field_data_declaration,
        ],
        name="particle_index_tendency_kernel",
    )


def test_kernel_function_impl_call_equivalent_to_manual(
    particle_property_args,
    scalar_args,
    field_data_args,
    particle_index_tendency_kernel_function,
    particle_index_tendency_kernel_function_implementation,
) -> None:
    """Test that the kernel function implementation created by the decorator is equivalent to a manually defined function."""
    # get the keys to pass to the decorator
    particle_property_keys = tuple(particle_property_args.keys())
    scalar_keys = tuple(scalar_args.keys())
    field_data_keys = tuple(field_data_args.keys())

    # call the original kernel function with the mock arguments
    particle_index_tendency_kernel_function(particle_property_args, scalar_args, field_data_args)
    manual_call_args = particle_index_tendency_kernel_function_implementation.call_args

    # reset the mock call history to isolate the calls for the decorated function
    particle_index_tendency_kernel_function_implementation.reset_mock()

    # get the kernel function implementation created by the decorator
    decorated_kernel_function = kernel_function_impl(
        particle_property_keys,
        scalar_keys,
        field_data_keys,
    )(particle_index_tendency_kernel_function_implementation)

    # call the decorated kernel function with the mock arguments
    decorated_kernel_function(particle_property_args, scalar_args, field_data_args)
    decorated_call_args = particle_index_tendency_kernel_function_implementation.call_args

    assert manual_call_args == decorated_call_args


def test_kernel_function_impl_preserves_name_and_docstring() -> None:
    """Test that the kernel function implementation created by the decorator preserves the name and docstring of the original function."""
    decorated_kernel_function = kernel_function_impl([], [], [])(
        _particle_index_tendency_kernel_function_implementation
    )

    assert decorated_kernel_function.__name__ == _particle_index_tendency_kernel_function_implementation.__name__  # type: ignore[attr-defined]
    assert decorated_kernel_function.__doc__ == _particle_index_tendency_kernel_function_implementation.__doc__


def test_particle_kernel_same_as_manual(
    particle_index_tendency_kernel,
    particle_index_tendency_kernel_function,
    index_particle_property_declaration,
    index_tendency_particle_property_declaration,
    timestep_scalar_declaration,
    velocity_field_data_declaration,
    metric_field_data_declaration,
) -> None:
    """Test that the ParticleKernel created by the decorator returns the same results as a manually defined function."""
    decorator_kernel = particle_kernel(
        [index_particle_property_declaration, index_tendency_particle_property_declaration],
        [timestep_scalar_declaration],
        [velocity_field_data_declaration, metric_field_data_declaration],
        name="particle_index_tendency_kernel",
    )(particle_index_tendency_kernel_function)

    # both kernels use the same underlying function so we should get a complete match down to the description
    assert decorator_kernel.functions == particle_index_tendency_kernel.functions
    assert decorator_kernel.particle_properties == particle_index_tendency_kernel.particle_properties
    assert decorator_kernel.scalars == particle_index_tendency_kernel.scalars
    assert decorator_kernel.field_data == particle_index_tendency_kernel.field_data
    assert decorator_kernel.description == particle_index_tendency_kernel.description


def test_particle_kernel_impl_equivalent_to_manual(
    particle_index_tendency_kernel,
    particle_index_tendency_kernel_function_implementation,
    index_particle_property_declaration,
    index_tendency_particle_property_declaration,
    timestep_scalar_declaration,
    velocity_field_data_declaration,
    metric_field_data_declaration,
) -> None:
    """Test that the ParticleKernel implementation created by the decorator returns the same results as a manually defined function."""
    decorator_kernel = particle_kernel_impl(
        [index_particle_property_declaration, index_tendency_particle_property_declaration],
        [timestep_scalar_declaration],
        [velocity_field_data_declaration, metric_field_data_declaration],
        name="particle_index_tendency_kernel",
    )(particle_index_tendency_kernel_function_implementation)

    # here the underlying functions are not the same object so only check the declarations match
    assert decorator_kernel.particle_properties == particle_index_tendency_kernel.particle_properties
    assert decorator_kernel.scalars == particle_index_tendency_kernel.scalars
    assert decorator_kernel.field_data == particle_index_tendency_kernel.field_data


def test_particle_kernel_impl_call_with_same_arguments_as_manual(
    particle_index_tendency_kernel,
    particle_index_tendency_kernel_function_implementation,
    index_particle_property_declaration,
    index_tendency_particle_property_declaration,
    timestep_scalar_declaration,
    velocity_field_data_declaration,
    metric_field_data_declaration,
    particle_property_args,
    scalar_args,
    field_data_args,
) -> None:
    """Test that the decorator and manual ParticleKernel implementations call the underlying function with the same arguments."""
    decorator_kernel = particle_kernel_impl(
        [index_particle_property_declaration, index_tendency_particle_property_declaration],
        [timestep_scalar_declaration],
        [velocity_field_data_declaration, metric_field_data_declaration],
        name="particle_index_tendency_kernel",
    )(particle_index_tendency_kernel_function_implementation)

    # call the decorated kernel implementation with the mock arguments
    decorator_kernel(particle_property_args, scalar_args, field_data_args)

    # check that the underlying function was called
    particle_index_tendency_kernel_function_implementation.assert_called_once()

    # reset the mock call history to isolate the calls for the original function
    particle_index_tendency_kernel_function_implementation.reset_mock()

    # call the original kernel function with the mock arguments
    particle_index_tendency_kernel(particle_property_args, scalar_args, field_data_args)

    # check that the underlying function was called with the same arguments
    particle_index_tendency_kernel_function_implementation.assert_called_once()


def test_particle_kernel_impl_accepts_generator_declarations(
    particle_index_tendency_kernel_function_implementation,
    index_particle_property_declaration,
    index_tendency_particle_property_declaration,
    timestep_scalar_declaration,
    velocity_field_data_declaration,
    metric_field_data_declaration,
):
    """Test that the particle_kernel_impl decorator can accept generator expressions, i.e. a single-pass iterator, for declarations."""
    decorator_kernel = particle_kernel_impl(
        (d for d in [index_particle_property_declaration, index_tendency_particle_property_declaration]),
        (d for d in [timestep_scalar_declaration]),
        (d for d in [velocity_field_data_declaration, metric_field_data_declaration]),
    )(particle_index_tendency_kernel_function_implementation)

    assert list(decorator_kernel.particle_properties.values()) == [
        index_particle_property_declaration,
        index_tendency_particle_property_declaration,
    ]
    assert list(decorator_kernel.scalars.values()) == [timestep_scalar_declaration]
    assert list(decorator_kernel.field_data.values()) == [
        velocity_field_data_declaration,
        metric_field_data_declaration,
    ]
