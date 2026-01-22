"""Submodule for declaring output."""

import abc
import dataclasses
import functools
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from ..events import Event, SimulationState
from ..kernels import ParticleKernel, merge_particle_fields


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class Output:
    """Class defining a single output."""

    particle_set: str
    particle_field: str
    kernels: tuple[ParticleKernel, ...]
    dtype: npt.DTypeLike
    attrs: dict[str, Any]

    def __init__(
        self,
        particle_set: str,
        particle_field: str,
        *kernels: ParticleKernel,
        dtype: npt.DTypeLike | None = None,
        **attrs: Any,
    ) -> None:
        """Initialize the Output."""

        if dtype is None:
            # infer dtype from kernels or default to float64
            fields = merge_particle_fields(kernels)
            dtype = fields.get(particle_field, np.float64)

        object.__setattr__(self, "particle_set", particle_set)
        object.__setattr__(self, "particle_field", particle_field)
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "attrs", dict(attrs))

    @property
    def required_fields(self) -> set[str]:
        """Get the particle fields required by the output."""
        fields = set(self.particle_field)
        for kernel in self.kernels:
            fields.update(kernel.particle_fields)
        return fields


class AbstractOutputWriter(abc.ABC):
    """Interface for output writers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the output writer."""
        pass

    @property
    @abc.abstractmethod
    def outputs(self) -> Mapping[str, Output]:
        """The outputs declared for this writer."""
        pass

    @abc.abstractmethod
    def write_time(self, state: SimulationState) -> None:
        """Write the current simulation time.

        Args:
            state: The current simulation state.
        """
        pass

    @abc.abstractmethod
    def write_output(self, key: str, state: SimulationState) -> None:
        """Write output for a given variable at the current time step.

        Args:
            key: The identifier of the output variable to write.
            state: The current simulation state.
        """
        pass

    @abc.abstractmethod
    def finalise_write_round(self, state: SimulationState) -> None:
        """Confirm that all outputs have been written for the current round."""
        pass

    def event_name(self, output_name: str) -> str:
        """Generate an event name for an output.

        Args:
            output_name: The name of the output variable.
        """
        return f"{self.name}:{output_name}"

    def create_events(self) -> list[Event]:
        """Create events for writing output.

        Returns:
            A list of events for writing output.
        """

        events = []

        # write time
        time_event = Event(self.event_name("time"), self.write_time)
        events.append(time_event)

        # write outputs
        for key, output in self.outputs.items():
            event_func = functools.partial(self.write_output, key)
            event = Event(self.event_name(key), event_func, **{output.particle_set: output.kernels})
            events.append(event)

        # finalise write round
        finalise_write_round_event = Event(self.event_name("finalise"), self.finalise_write_round)
        events.append(finalise_write_round_event)

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
    def outputs(self) -> Mapping[str, Output]:
        """The outputs declared for this writer."""
        pass

    @abc.abstractmethod
    def add_output(self, key, output: Output, **kwargs: Any) -> None:
        """Add an output to the writer.

        Args:
            key: The identifier for the output.
            outputs: The output to add.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abc.abstractmethod
    def remove_output(self, key: str) -> None:
        """Remove an output from the writer.

        Args:
            key: The identifier of the output to remove.
        """
        pass

    @abc.abstractmethod
    def build(
        self,
        nparticles: dict[str, int],
        time_type: np.dtype,
    ) -> AbstractOutputWriter:
        """Build the output writer.

        Args:
            nparticles: A mapping of particle set names to number of particles.
            time_type: The numpy dtype for the time variable.
        """
        pass
