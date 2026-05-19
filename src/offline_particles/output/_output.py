"""Submodule for declaring output."""

import abc
import collections.abc
import dataclasses
import functools
import types
from typing import Any, Iterable, KeysView, Mapping

import numpy as np
import numpy.typing as npt

from ..events import Event, SimulationState
from ..kernels import BoundKernel, ParticlePropertyDeclaration, get_required_particle_property_dtypes


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class Output:
    """Class defining a single output."""

    particle_property: ParticlePropertyDeclaration
    kernels: tuple[BoundKernel, ...]
    attrs: dict[str, Any]

    def __init__(
        self,
        particle_property_name: str,
        *kernels: BoundKernel,
        dtype: npt.DTypeLike | None = None,
        **attrs: Any,
    ) -> None:
        """Initialize the Output.

        Parameters
        ----------
        particle_property_name : str
            The name of the particle property to output.
        *kernels : BoundKernel
            The kernels required to compute the output.
        dtype : npt.DTypeLike, optional
            The data type of the output.
        **attrs : Any
            Additional attributes for the output.

        Raises
        ------
        ValueError
            If the specified dtype is incompatible with the required dtype from the kernels.
        """
        kernel_particle_property_dtypes = get_required_particle_property_dtypes(*kernels)

        # case 1: the particle property is already defined by the kernels
        if particle_property_name in kernel_particle_property_dtypes:
            particle_property_dtype = kernel_particle_property_dtypes[particle_property_name]
            # Check dtype compatibility
            if dtype is not None and np.dtype(dtype) != particle_property_dtype:
                raise ValueError(
                    f"Output particle property '{particle_property_name}' has dtype "
                    f"{dtype}, but a different dtype {particle_property_dtype} is required by the kernels."
                )
        # case 2: the particle property not required by kernels
        else:
            # default value for dtype
            if dtype is None:
                dtype = np.float64
            particle_property_dtype = np.dtype(dtype)
        particle_property = ParticlePropertyDeclaration(particle_property_name, particle_property_dtype)

        object.__setattr__(self, "particle_property", particle_property)
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "attrs", dict(attrs))

    @property
    def required_property_dtypes(self) -> Mapping[str, np.dtype]:
        """Get the particle properties dtypes required by the output."""
        required_property_dtypes = {self.particle_property.name: self.particle_property.dtype}
        required_property_dtypes.update(get_required_particle_property_dtypes(*self.kernels))
        return types.MappingProxyType(required_property_dtypes)


class TwoKeyDict[OT, IT, VT](collections.abc.MutableMapping[tuple[OT, IT], VT]):
    """A dictionary-like class that uses two keys to identify values."""

    def __init__(self) -> None:
        self._data: dict[OT, dict[IT, VT]] = {}

    def __getitem__(self, keys: tuple[OT, IT]) -> VT:
        outer, inner = keys
        return self._data[outer][inner]

    def __setitem__(self, keys: tuple[OT, IT], value: VT) -> None:
        outer, inner = keys
        if outer not in self._data:
            self._data[outer] = {}
        self._data[outer][inner] = value

    def __delitem__(self, keys: tuple[OT, IT]) -> None:
        outer, inner = keys
        del self._data[outer][inner]
        if not self._data[outer]:  # Remove the outer key if the inner dict is empty
            del self._data[outer]

    def __iter__(self):
        for outer, inner_dict in self._data.items():
            for inner in inner_dict:
                yield (outer, inner)

    def __len__(self) -> int:
        return sum(len(inner_dict) for inner_dict in self._data.values())

    def get_inner_mapping(self, outer_key: OT) -> Mapping[IT, VT]:
        """Get a view of the inner mapping for a given outer key.

        Parameters
        ----------
        outer_key : OT
            The outer key for which to retrieve the inner mapping.

        Returns
        -------
        Mapping[IT, VT]
            A read-only view of the inner mapping corresponding to the specified outer key.

        Raises
        ------
        KeyError
            If the specified outer key does not exist in the TwoKeyDict.
        """
        if outer_key not in self._data:
            raise KeyError(f"Outer key '{outer_key}' not found.")
        return types.MappingProxyType(self._data[outer_key])

    def outer_keys(self) -> KeysView[OT]:
        """Get a view of all outer keys.

        Returns
        -------
        KeysView[OT]
            A view of all outer keys in the TwoKeyDict.
        """
        return self._data.keys()


class AbstractOutputWriter(abc.ABC):
    """Interface for output writers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the output writer."""
        pass

    @property
    @abc.abstractmethod
    def outputs(self) -> Iterable[tuple[tuple[str, str], Output]]:
        """The outputs declared for this writer."""
        pass

    @property
    @abc.abstractmethod
    def static_outputs(self) -> Iterable[tuple[tuple[str, str], Output]]:
        """The static (time-independent) outputs declared for this writer.

        Static outputs are written once at iteration 0, after particle initialisation.
        """
        pass

    @abc.abstractmethod
    def write_time(self, state: SimulationState) -> None:
        """Write the current simulation time.

        Args:
            state: The current simulation state.
        """
        pass

    @abc.abstractmethod
    def write_output(self, particle_set: str, name: str, state: SimulationState) -> None:
        """Write output for a given variable at the current time step.

        Args:
            particle_set: The set of particles for which to write output.
            name: The name of the output variable to write.
            state: The current simulation state.
        """
        pass

    @abc.abstractmethod
    def finalise_write_round(self, state: SimulationState) -> None:
        """Confirm that all outputs have been written for the current round."""
        pass

    @abc.abstractmethod
    def write_static_output(self, particle_set: str, name: str, state: SimulationState) -> None:
        """Write a static (time-independent) output variable once.

        This is called at iteration 0, after particle initialisation.

        Args:
            particle_set: The set of particles for which to write the static output.
            name: The name of the static output variable to write.
            state: The current simulation state.
        """
        pass

    def event_name(self, particle_set: str, name: str) -> str:
        """Generate an event name for an output.

        Parameters
        ----------
        particle_set : str
            The set of particles for which to write output.
        name : str
            The name of the output variable.

        Returns
        -------
        str
            The generated event name in the format "{writer_name}:{particle_set}:{output_name}".
        """
        return f"{self.name}:{particle_set}:{name}"

    def create_output_events(self) -> list[Event]:
        """Create recurring events for writing time-dependent output.

        Returns
        -------
            A list of recurring events for writing time-dependent output.
        """
        events = []

        # write time
        time_event = Event(f"{self.name}:time", self.write_time)
        events.append(time_event)

        # write outputs
        for (particle_set, name), output in self.outputs:
            event_func = functools.partial(self.write_output, particle_set, name)
            event = Event(self.event_name(particle_set, name), event_func, **{particle_set: output.kernels})
            events.append(event)

        # finalise write round
        finalise_write_round_event = Event(f"{self.name}:finalise", self.finalise_write_round)
        events.append(finalise_write_round_event)

        return events

    def create_static_output_events(self) -> list[Event]:
        """Create one-shot events for writing static (time-independent) outputs.

        These events are intended to be registered once at iteration 0,
        after particle initialisation.

        Returns
        -------
            A list of one-shot events for writing static outputs.
        """
        events = []
        for (particle_set, name), output in self.static_outputs:
            event_func = functools.partial(self.write_static_output, particle_set, name)
            event = Event(self.event_name(particle_set, name), event_func, **{particle_set: output.kernels})
            events.append(event)
        return events


class AbstractOutputWriterBuilder(abc.ABC):
    """Abstract base class for output writer builders."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the output writer."""
        pass

    @property
    @abc.abstractmethod
    def outputs(self) -> Iterable[tuple[tuple[str, str], Output]]:
        """The outputs declared for this writer."""
        pass

    @property
    @abc.abstractmethod
    def static_outputs(self) -> Iterable[tuple[tuple[str, str], Output]]:
        """The static (time-independent) outputs declared for this writer.

        Note:
            Static outputs are written once at iteration 0, after particle initialisation.
        """
        pass

    @abc.abstractmethod
    def add_output(self, particle_set: str, name: str, output: Output, **kwargs: Any) -> None:
        """Add an output to the writer.

        Args:
            particle_set: The set of particles for which to add the output.
            name: The name of the output.
            output: The output to add.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abc.abstractmethod
    def remove_output(self, particle_set: str, name: str) -> None:
        """Remove an output from the writer.

        Args:
            particle_set: The set of particles for which to remove the output.
            name: The name of the output to remove.
        """
        pass

    @abc.abstractmethod
    def add_static_output(self, particle_set: str, name: str, output: Output, **kwargs: Any) -> None:
        """Add a static (time-independent) output to the writer.

        Static outputs are written once at iteration 0, after particle initialisation.

        Args:
            particle_set: The set of particles for which to add the static output.
            name: The name of the static output.
            output: The output to add.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abc.abstractmethod
    def remove_static_output(self, particle_set: str, name: str) -> None:
        """Remove a static output from the writer.

        Args:
            particle_set: The set of particles for which to remove the static output.
            name: The name of the static output to remove.
        """
        pass

    @abc.abstractmethod
    def build(
        self,
        nparticles: dict[str, int],
        time_type: npt.DTypeLike,
    ) -> AbstractOutputWriter:
        """Build the output writer.

        Args:
            nparticles: A mapping of particle set names to number of particles.
            time_type: The numpy dtype for the time variable.
        """
        pass
