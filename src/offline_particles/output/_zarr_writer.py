"""Write output to Zarr stores."""

import dataclasses
import itertools
import types
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr
import zarr.storage

from ..events import SimulationState
from ._output import AbstractOutputWriter, AbstractOutputWriterBuilder, Output, TwoKeyDict

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
        time_arrays: Mapping[str, zarr.Array],
        outputs: TwoKeyDict[str, str, ZarrOutputArray],
        static_outputs: TwoKeyDict[str, str, ZarrOutputArray],
    ) -> None:
        """Initialize the Zarr output writer.

        Parameters
        ----------
        name : str
            The name of the output writer.
        store : zarr.storage.StoreLike
            The Zarr store to write to.
        time_arrays : Mapping[str, zarr.Array]
            A dictionary mapping particle sets to Zarr arrays for time output.
        outputs : TwoKeyDict[str, str, ZarrOutputArray]
            A two-key mapping from particle set names and output names to ZarrOutputArrays for time-dependent outputs.
        static_outputs : TwoKeyDict[str, str, ZarrOutputArray]
            A two-key mapping from particle set names and static output names to ZarrOutputArrays for static outputs.
        """
        self._name = name
        self._store = store
        self._time_arrays = types.MappingProxyType(time_arrays)
        self._outputs = outputs
        self._static_outputs = static_outputs
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
    def outputs(self) -> Iterable[tuple[tuple[str, str], Output]]:
        """The outputs declared for this writer.

        Yields
        ------
        key : tuple[str, str]
            The key (particle_set, name)
        output : Output
            The corresponding Output object.
        """
        for key, zarr_output_array in self._outputs.items():
            yield key, zarr_output_array.output

    @property
    def static_outputs(self) -> Iterable[tuple[tuple[str, str], Output]]:
        """The static (time-independent) outputs declared for this writer.

        Yields
        ------
        key : tuple[str, str]
            The key (particle_set, name)
        output : Output
            The corresponding Output object.
        """
        for key, zarr_output_array in self._static_outputs.items():
            yield key, zarr_output_array.output

    def write_time(self, state: SimulationState) -> None:
        """Write the current simulation time.

        Args:
            state: The current simulation state; ``state.time`` is written.

        Note:
            Each particle set group has its own time array.
        """
        for array in self._time_arrays.values():
            array.append(np.array([state.time]), axis=0)

    def write_output(self, particle_set: str, name: str, state: SimulationState) -> None:
        """Write output for a given variable at the current time step.

        Parameters
        ----------
        particle_set : str
            The set of particles for which to write output.
        name : str
            The name of the output variable to write.
        state : SimulationState
            The current simulation state.

        Raises
        ------
        KeyError
            If the output variable does not exist.
        """
        key = (particle_set, name)
        if key not in self._outputs:
            raise KeyError(f"Output variable '{key}' not found.")

        zarr_output_array = self._outputs[key]
        output = zarr_output_array.output
        array = zarr_output_array.array
        particle_property = output.particle_property

        # write output
        time_size, particle_size = array.shape
        array.resize((time_size + 1, particle_size))
        array[-1, :] = state.particles[particle_set][particle_property]

    def write_static_output(self, particle_set: str, name: str, state: SimulationState) -> None:
        """Write a static (time-independent) output variable once.

        Parameters
        ----------
        particle_set : str
            The set of particles for which to write the static output.
        name : str
            The name of the static output variable to write.
        state : SimulationState
            The current simulation state.

        Raises
        ------
        KeyError
            If the static output variable does not exist.

        Notes
        -----
        This is called at iteration 0, after particle initialisation.
        """
        key = (particle_set, name)
        if key not in self._static_outputs:
            raise KeyError(f"Static output variable '{key}' not found.")

        zarr_output_array = self._static_outputs[key]
        output = zarr_output_array.output
        array = zarr_output_array.array
        particle_property = output.particle_property

        array[:] = state.particles[particle_set][particle_property]

    def finalise_write_round(self, state: SimulationState) -> None:
        """Confirm that all outputs have been written for the current round and then increments the count.

        Parameters
        ----------
        state : SimulationState
            The current simulation state.

        Raises
        ------
        RuntimeError
            If the number of time entries in any output array does not match the expected count.
        """
        expected_count = self._output_count + 1

        # check time output
        for particle_set, array in self._time_arrays.items():
            time_count = array.shape[0]
            if time_count != expected_count:
                raise RuntimeError(
                    f"Time output in group '{particle_set}' has {time_count} entries, expected {expected_count}."
                )

        # check all other outputs
        for (particle_set, name), zarr_output_array in self._outputs.items():
            if zarr_output_array.array.shape[0] != expected_count:
                raise RuntimeError(
                    f"Output '{name}' for group '{particle_set}' has {zarr_output_array.array.shape[0]} time entries, expected {expected_count}."
                )

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
        time_name: str = "time",
        overwrite: bool = False,
        array_kwargs: dict[str, Any] | None = None,
        time_array_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Zarr output writer builder.

        Parameters
        ----------
        name : str
            The name of the output writer to build.
        store : zarr.storage.StoreLike
            The Zarr store to write to.
        chunksize : int, optional
            The chunk size for the particle dimension.
        time_name : str, optional
            The name of the time output array.
        overwrite : bool, optional
            Whether to overwrite existing data in the store.
        array_kwargs : dict[str, Any] | None, optional
            Default keyword arguments passed to Zarr.create_array for all outputs.
        time_array_kwargs : dict[str, Any] | None, optional
            Keyword arguments passed, in addition to array_kwargs, to Zarr.create_array for the time array.
        """
        self._name = name
        self._store = store
        self._outputs: TwoKeyDict[str, str, ZarrOutputDefinition] = TwoKeyDict()
        self._static_outputs: TwoKeyDict[str, str, ZarrOutputDefinition] = TwoKeyDict()

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
    def outputs(self) -> Iterable[tuple[tuple[str, str], Output]]:
        """The outputs declared for this writer.

        Yields
        ------
        tuple[str, str]
            The key (particle_set, name)
        Output
            The corresponding Output object.
        """
        for key, zarr_output_def in self._outputs.items():
            yield key, zarr_output_def.output

    @property
    def static_outputs(self) -> Iterable[tuple[tuple[str, str], Output]]:
        """The static (time-independent) outputs declared for this writer.

        Yields
        ------
        tuple[str, str]
            The key (particle_set, name)
        Output
            The corresponding Output object.
        """
        for key, zarr_output_def in self._static_outputs.items():
            yield key, zarr_output_def.output

    def add_output(self, particle_set: str, name: str, output: Output, **kwargs) -> None:
        """Add output to the writer.

        Parameters
        ----------
        particle_set : str
            The particle set to which the output belongs.
        name : str
            The name of the output. Also used as the Zarr array name unless 'name' is given in kwargs.
        output : Output
            The output to add.
        **kwargs : Any
            Additional keyword arguments passed to Zarr.create_array for this output.

        Raises
        ------
        KeyError
            If the output variable already exists or if the name is already used by a static output.
        """
        key = (particle_set, name)
        array_kwargs = self._array_kwargs.copy()
        array_kwargs.update(kwargs)

        if key in self._outputs:
            raise KeyError(f"Output variable with key '{key}' already exists.")
        if key in self._static_outputs:
            raise KeyError(f"Output key '{key}' is already used by a static output.")

        self._outputs[key] = ZarrOutputDefinition(output, array_kwargs)

    def remove_output(self, particle_set: str, name: str) -> None:
        """Remove an output from the writer.

        Parameters
        ----------
        particle_set : str
            The particle set to which the output belongs.
        name : str
            The name of the output to remove.

        Raises
        ------
        KeyError
            If the output variable does not exist.
        """
        key = (particle_set, name)
        if key not in self._outputs:
            raise KeyError(f"Output variable '{key}' does not exist.")

        del self._outputs[key]

    def add_static_output(self, particle_set: str, name: str, output: Output, **kwargs) -> None:
        """Add a static (time-independent) output to the writer.

        Parameters
        ----------
        particle_set : str
            The particle set to which the static output belongs.
        name : str
            The name of the static output. Also used as the Zarr array name unless 'name' is given in kwargs.
        output : Output
            The output to add.
        **kwargs : Any
            Additional keyword arguments passed to Zarr.create_array for this output.

        Raises
        ------
        KeyError
            If the static output variable already exists or if the name is already used by a time-dependent output.

        Notes
        -----
        Static outputs are written once at iteration 0, after particle initialisation.
        """
        key = (particle_set, name)
        array_kwargs = self._array_kwargs.copy()
        array_kwargs.update(kwargs)

        if key in self._static_outputs:
            raise KeyError(f"Static output variable with key '{key}' already exists.")
        if key in self._outputs:
            raise KeyError(f"Output key '{key}' is already used by a time-dependent output.")

        self._static_outputs[key] = ZarrOutputDefinition(output, array_kwargs)

    def remove_static_output(self, particle_set: str, name: str) -> None:
        """Remove a static output from the writer.

        Parameters
        ----------
        particle_set : str
            The particle set to which the static output belongs.
        name : str
            The name of the static output to remove.

        Raises
        ------
        KeyError
            If the static output variable does not exist.
        """
        key = (particle_set, name)
        if key not in self._static_outputs:
            raise KeyError(f"Static output variable '{key}' does not exist.")

        del self._static_outputs[key]

    def build(self, nparticles: dict[str, int], time_type: npt.DTypeLike = np.float64) -> ZarrOutputWriter:
        # validate particle_sets
        for particle_set in itertools.chain(self._outputs.outer_keys(), self._static_outputs.outer_keys()):
            if particle_set not in nparticles:
                raise KeyError(f"Number of particles for particle set '{particle_set}' not provided.")

        # initialise time array for each particle set group
        time_arrays = {
            particle_set: zarr.create_array(
                self._store,
                name=f"{particle_set}/{self._time_name}",
                shape=(0,),
                dtype=np.dtype(time_type),
                chunks=(1,),
                dimension_names=(self._time_name,),
                overwrite=self._overwrite,
                **self._time_array_kwargs,
            )
            for particle_set in nparticles
        }

        # create output arrays
        outputs = TwoKeyDict()
        for (particle_set, name), zarr_output_def in self._outputs.items():
            output = zarr_output_def.output
            kwargs = zarr_output_def.kwargs.copy()
            num_particles = nparticles[particle_set]

            # create output array
            array_name = kwargs.pop("name", name)
            outputs[particle_set, name] = ZarrOutputArray(
                output,
                self._initialize_output_array(particle_set, array_name, output, num_particles, kwargs),
            )

        # create static output arrays (1D, written once)
        static_outputs = TwoKeyDict()
        for (particle_set, name), zarr_output_def in self._static_outputs.items():
            output = zarr_output_def.output
            kwargs = zarr_output_def.kwargs.copy()

            # get nparticles for this particle set
            num_particles = nparticles[particle_set]

            # create static output array
            array_name = kwargs.pop("name", name)
            static_outputs[particle_set, name] = ZarrOutputArray(
                output,
                self._initialize_static_output_array(particle_set, array_name, output, num_particles, kwargs),
            )

        return ZarrOutputWriter(
            name=self._name,
            store=self._store,
            time_arrays=time_arrays,
            outputs=outputs,
            static_outputs=static_outputs,
        )

    def _initialize_output_array(
        self, particle_set: str, name: str, output: Output, nparticles: int, array_kwargs: dict[str, Any]
    ) -> zarr.Array:
        """Initialize Zarr array for output.

        Parameters
        ----------
        particle_set : str
            The name of the particle set.
        name : str
            The name of the output variable.
        output : Output
            The output definition.
        nparticles : int
            The number of particles in the particle set.
        array_kwargs : dict[str, Any]
            Additional keyword arguments passed to Zarr.create_array.

        Returns
        -------
        zarr.Array
            The created Zarr array for the output.
        """
        # set shape and chunks
        shape = (0, nparticles)
        chunks = (1, min(self._chunksize, nparticles))

        # create array
        array = zarr.create_array(
            self._store,
            name=f"{particle_set}/{name}",
            shape=shape,
            dtype=output.dtype,
            chunks=chunks,
            attributes=output.attrs,
            dimension_names=(self._time_name, particle_set),
            overwrite=self._overwrite,
            **array_kwargs,
        )
        return array

    def _initialize_static_output_array(
        self, particle_set: str, name: str, output: Output, nparticles: int, array_kwargs: dict[str, Any]
    ) -> zarr.Array:
        """Initialize Zarr array for a static (time-independent) output.

        Parameters
        ----------
        particle_set : str
            The name of the particle set.
        name : str
            The name of the output variable.
        output : Output
            The output definition.
        nparticles : int
            The number of particles in the particle set.
        array_kwargs : dict[str, Any]
            Additional keyword arguments passed to Zarr.create_array.

        Returns
        -------
        zarr.Array
            The created Zarr array for the static output.
        """
        # set shape and chunks (1D: particles only)
        shape = (nparticles,)
        chunks = (max(1, min(self._chunksize, nparticles)),)

        # create array
        array = zarr.create_array(
            self._store,
            name=f"{particle_set}/{name}",
            shape=shape,
            dtype=output.dtype,
            chunks=chunks,
            attributes=output.attrs,
            dimension_names=(particle_set,),
            overwrite=self._overwrite,
            **array_kwargs,
        )
        return array
