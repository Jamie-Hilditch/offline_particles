"""Write output to Zarr stores."""

import types
from typing import Any, Mapping

import numpy as np
import zarr
import zarr.storage

from ..events import AbstractSchedule, SimulationState
from ._output import AbstractOutputWriter, AbstractOutputWriterBuilder, Output

DEFAULT_CHUNKSIZE = 10_000


class ZarrOutputWriter(AbstractOutputWriter):
    """Class for writing output to Zarr format."""

    def __init__(
        self,
        name: str,
        schedule: AbstractSchedule,
        store: zarr.storage.StoreLike,
        time_array: zarr.Array,
        outputs: dict[str, tuple[Output, zarr.Array]],
    ) -> None:
        """Initialize the Zarr output writer.

        Args:
            store: The Zarr store to write to.
            time_array: The Zarr array for time output.
            outputs: A dictionary mapping output names to Output.
        """
        self._name = name
        self._store = store
        self._schedule = schedule
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
    def schedule(self) -> AbstractSchedule:
        """The output schedule."""
        return self._schedule

    @property
    def outputs(self) -> Mapping[str, Output]:
        """The outputs declared for this writer."""
        return types.MappingProxyType({key: output for key, (output, _) in self._outputs.items()})

    def write_time(self, state: SimulationState) -> None:
        """Write the current simulation time.

        Args:
            time: The current simulation time.
        """
        self._time_array.append(np.array([state.time]), axis=0)

    def write_output(self, name: str, state: SimulationState) -> None:
        """Write output for a given variable at the current time step.

        Args:
            name: The name of the output variable to write.
            particles: The current view of the particles.
            time: The current simulation time.
        """
        if name not in self._outputs:
            raise KeyError(f"Output variable '{name}' not found.")

        output, output_array = self._outputs[name]
        field = output.particle_field

        # write output
        time_size, particle_size = output_array.shape
        output_array.resize((time_size + 1, particle_size))
        output_array[-1, :] = state.particles[field]

    def finalise_write_round(self, state: SimulationState) -> None:
        """Confirm that all outputs have been written for the current round and then increments the count."""
        expected_count = self._output_count + 1

        # check time output
        time_count = self._time_array.shape[0]
        if time_count != expected_count:
            raise RuntimeError(f"Time output has {time_count} entries, expected {expected_count}.")

        # check all other outputs
        for name, (_, output_array) in self._outputs.items():
            if output_array.shape[0] != expected_count:
                raise RuntimeError(
                    f"Output '{name}' has {output_array.shape[0]} time entries, expected {expected_count}."
                )

        # increment count
        self._output_count += 1


class ZarrOutputBuilder(AbstractOutputWriterBuilder):
    """Builder for zarr output."""

    def __init__(
        self,
        name: str,
        schedule: AbstractSchedule,
        store: zarr.storage.StoreLike,
        *,
        chunksize: int = DEFAULT_CHUNKSIZE,
        particle_dimension_name: str = "particle",
        time_name: str = "time",
        overwrite: bool = False,
        array_kwargs: dict[str, Any] | None = None,
        time_array_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Zarr output writer builder.

        Args:
            store: The Zarr store to write to.
            schedule: The output schedule.

        Keywords:
            chunksize: The chunk size for the particle dimension.
            particle_dimension_name: The name of the particle dimension.
            time_name: The name of the time output array.
            overwrite: Whether to overwrite existing data in the store.
            array_kwargs: Default keyword arguments passed to Zarr.create_array for all outputs.
            time_array_kwargs: Keyword arguments passed, in addition to array_kwargs, to Zarr.create_array for the time array.
        """
        self._name = name
        self._schedule = schedule
        self._store = store
        self._outputs: dict[str, tuple[Output, dict[str, Any]]] = {}

        self._chunksize = chunksize
        self._particle_dimension_name = particle_dimension_name
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
    def schedule(self) -> AbstractSchedule:
        """The output schedule."""
        return self._schedule

    @property
    def outputs(self) -> Mapping[str, Output]:
        """The outputs declared for this writer."""
        return types.MappingProxyType({key: output for key, (output, _) in self._outputs.items()})

    def add_output(self, *outputs: Output, **kwargs) -> None:
        """Add outputs to the writer.

        Args:
            *outputs: The outputs to add.
            **kwargs: Additional keyword arguments passed to Zarr.create_array for these outputs.
        """
        array_kwargs = self._array_kwargs.copy()
        array_kwargs.update(kwargs)

        for output in outputs:
            name = output.name
            if name in self._outputs:
                raise KeyError(f"Output variable with name '{name}' already exists.")

            self._outputs[name] = (output, array_kwargs)

    def remove_output(self, name: str) -> None:
        """Remove an output from the writer.

        Args:
            name: The name of the output to remove.
        """
        if name not in self._outputs:
            raise KeyError(f"Output variable '{name}' does not exist.")

        del self._outputs[name]

    def build(
        self,
        nparticles: int,
    ) -> ZarrOutputWriter:
        # initialise outputs
        time_output = zarr.create_array(
            self._store,
            name=self._time_name,
            shape=(0,),
            dtype="f8",
            chunks=(1,),
            dimension_names=(self._time_name,),
            overwrite=self._overwrite,
            **self._time_array_kwargs,
        )
        outputs = {
            name: (
                output,
                self._initialize_output_array(name, output, nparticles, array_kwargs),
            )
            for name, (output, array_kwargs) in self._outputs.items()
        }
        return ZarrOutputWriter(
            name=self._name,
            schedule=self._schedule,
            store=self._store,
            time_array=time_output,
            outputs=outputs,
        )

    def _initialize_output_array(
        self, name: str, output: Output, nparticles: int, array_kwargs: dict[str, Any]
    ) -> zarr.Array:
        """Initialize Zarr array for output."""
        shape = (0, nparticles)
        chunks = (1, min(self._chunksize, nparticles))
        return zarr.create_array(
            self._store,
            name=name,
            shape=shape,
            dtype=output.dtype,
            chunks=chunks,
            dimension_names=(self._time_name, self._particle_dimension_name),
            overwrite=self._overwrite,
            **array_kwargs,
        )
