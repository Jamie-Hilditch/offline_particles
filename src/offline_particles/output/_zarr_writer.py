"""Write output to Zarr stores."""

import dataclasses
import types
from typing import Any, Mapping

import numpy as np
import zarr
import zarr.storage

from ..events import SimulationState
from ._output import AbstractOutputWriter, AbstractOutputWriterBuilder, Output

DEFAULT_CHUNKSIZE = 250_000


@dataclasses.dataclass(slots=True)
class ZarrOutputArray:
    """Class representing a Zarr output array."""

    output: Output
    array: zarr.Array


@dataclasses.dataclass(slots=True)
class ZarrOutputDefinition:
    """Class representing a Zarr output definition."""

    output: Output
    kwargs: dict[str, Any]


class ZarrOutputWriter(AbstractOutputWriter):
    """Class for writing output to Zarr format."""

    def __init__(
        self,
        name: str,
        store: zarr.storage.StoreLike,
        time_array: zarr.Array,
        outputs: dict[str, ZarrOutputArray],
    ) -> None:
        """Initialize the Zarr output writer.

        Args:
            store: The Zarr store to write to.
            time_array: The Zarr array for time output.
            outputs: A dictionary mapping particle set names to lists of ZarrOutputArrays.
        """
        self._name = name
        self._store = store
        self._time_array = time_array
        self._outputs = types.MappingProxyType(outputs)
        self._output_count: int = 0

    @property
    def name(self) -> str:
        """The name of the output writer."""
        return self._name

    @property
    def store(self) -> zarr.storage.StoreLike:
        """The Zarr store."""
        return self._store

    @property
    def outputs(self) -> Mapping[str, Output]:
        """The outputs declared for this writer."""
        return types.MappingProxyType({key: zoa.output for key, zoa in self._outputs.items()})

    def write_time(self, state: SimulationState) -> None:
        """Write the current simulation time.

        Args:
            time: The current simulation time.
        """
        self._time_array.append(np.array([state.time]), axis=0)

    def write_output(self, key: str, state: SimulationState) -> None:
        """Write output for a given variable at the current time step.

        Args:
            key: The identifier of the output variable to write.
            state: The current simulation state.
        """
        if key not in self._outputs:
            raise KeyError(f"Output variable '{key}' not found.")

        zarr_output_array = self._outputs[key]
        output = zarr_output_array.output
        array = zarr_output_array.array
        field = output.particle_field

        # write output
        time_size, particle_size = array.shape
        array.resize((time_size + 1, particle_size))
        array[-1, :] = state.particles[output.particle_set][field]

    def finalise_write_round(self, state: SimulationState) -> None:
        """Confirm that all outputs have been written for the current round and then increments the count."""
        expected_count = self._output_count + 1

        # check time output
        time_count = self._time_array.shape[0]
        if time_count != expected_count:
            raise RuntimeError(f"Time output has {time_count} entries, expected {expected_count}.")

        # check all other outputs
        for name, zoa in self._outputs.items():
            if zoa.array.shape[0] != expected_count:
                raise RuntimeError(f"Output '{name}' has {zoa.array.shape[0]} time entries, expected {expected_count}.")

        # increment count
        self._output_count += 1


class ZarrOutputBuilder(AbstractOutputWriterBuilder):
    """Builder for zarr output."""

    def __init__(
        self,
        name: str,
        store: zarr.storage.StoreLike,
        *,
        chunksize: int = DEFAULT_CHUNKSIZE,
        consolidate_metadata: bool = True,
        time_name: str = "time",
        overwrite: bool = False,
        array_kwargs: dict[str, Any] | None = None,
        time_array_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Zarr output writer builder.

        Args:
            store: The Zarr store to write to.

        Keywords:
            chunksize: The chunk size for the particle dimension.
            time_name: The name of the time output array.
            overwrite: Whether to overwrite existing data in the store.
            array_kwargs: Default keyword arguments passed to Zarr.create_array for all outputs.
            time_array_kwargs: Keyword arguments passed, in addition to array_kwargs, to Zarr.create_array for the time array.
        """
        self._name = name
        self._store = store
        self._outputs: dict[str, ZarrOutputDefinition] = {}

        self._chunksize = chunksize
        self._time_name = time_name
        self._overwrite = overwrite
        if array_kwargs is None:
            array_kwargs = {}
        self._array_kwargs = array_kwargs
        self._time_array_kwargs = array_kwargs.copy()
        if time_array_kwargs is not None:
            self._time_array_kwargs.update(time_array_kwargs)

    @property
    def name(self) -> str:
        """The name of the output writer."""
        return self._name

    @property
    def outputs(self) -> Mapping[str, Output]:
        """The outputs declared for this writer."""
        return types.MappingProxyType({key: zod.output for key, zod in self._outputs.items()})

    def add_output(self, key: str, output: Output, **kwargs) -> None:
        """Add output to the writer.

        Args:
            key: The identifier for the output. Also used as the Zarr array name unless 'name' is given in kwargs.
            output: The output to add.
            **kwargs: Additional keyword arguments passed to Zarr.create_array for this output.
        """
        array_kwargs = self._array_kwargs.copy()
        array_kwargs.update(kwargs)

        if key in self._outputs:
            raise KeyError(f"Output variable with key '{key}' already exists.")

        self._outputs[key] = ZarrOutputDefinition(output, array_kwargs)

    def remove_output(self, key: str) -> None:
        """Remove an output from the writer.

        Args:
            key: The identifier of the output to remove.
        """
        if key not in self._outputs:
            raise KeyError(f"Output variable '{key}' does not exist.")

        del self._outputs[key]

    def build(self, nparticles: dict[str, int], time_type: np.dtype = np.dtype(np.float64)) -> ZarrOutputWriter:
        # initialise outputs
        time_output = zarr.create_array(
            self._store,
            name=self._time_name,
            shape=(0,),
            dtype=time_type,
            chunks=(1,),
            dimension_names=(self._time_name,),
            overwrite=self._overwrite,
            **self._time_array_kwargs,
        )

        # create output arrays
        outputs = {}
        for key, zod in self._outputs.items():
            output = zod.output
            kwargs = zod.kwargs

            # get nparticles for this particle set
            if output.particle_set not in nparticles:
                raise KeyError(f"Number of particles for particle set '{output.particle_set}' not provided.")
            num_particles = nparticles[output.particle_set]

            # create output array
            array_name = kwargs.pop("name", key)
            outputs[key] = ZarrOutputArray(
                output,
                self._initialize_output_array(array_name, output, num_particles, kwargs),
            )

        return ZarrOutputWriter(
            name=self._name,
            store=self._store,
            time_array=time_output,
            outputs=outputs,
        )

    def _initialize_output_array(
        self, name: str, output: Output, nparticles: int, array_kwargs: dict[str, Any]
    ) -> zarr.Array:
        """Initialize Zarr array for output."""

        # set shape and chunks
        shape = (0, nparticles)
        chunks = (1, min(self._chunksize, nparticles))

        # create array
        array = zarr.create_array(
            self._store,
            name=name,
            shape=shape,
            dtype=output.particle_property.dtype,
            chunks=chunks,
            attributes=output.attrs,
            dimension_names=(self._time_name, output.particle_set),
            overwrite=self._overwrite,
            **array_kwargs,
        )
        return array
