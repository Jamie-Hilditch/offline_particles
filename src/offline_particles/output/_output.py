"""Submodule for declaring output."""

import abc
import dataclasses
import functools
from typing import Any, Mapping

import numpy as np

from ..events import Event, SimulationState
from ..kernels import ParticleKernel


@dataclasses.dataclass(frozen=True, slots=True, init=False)
class Output:
    """Class defining a single output."""

    name: str
    particle_set: str
    particle_field: str
    kernels: tuple[ParticleKernel, ...]
    attrs: dict[str, Any]

    def __init__(
        self,
        name: str,
        particle_set: str,
        *kernels: ParticleKernel,
        particle_field: str | None = None,
        **attrs: Any,
    ) -> None:
        """Initialize the Output."""
        # default value for particle_field
        if particle_field is None:
            particle_field = name

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "particle_set", particle_set)
        object.__setattr__(self, "particle_field", particle_field)
        object.__setattr__(self, "kernels", kernels)
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
    def write_output(self, name: str, state: SimulationState) -> None:
        """Write output for a given variable at the current time step.

        Args:
            name: The complete name of the output variable to write.
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
        for name, output in self.outputs.items():
            event_func = functools.partial(self.write_output, name)
            event = Event(self.event_name(name), event_func, **{output.particle_set: output.kernels})
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
    def add_output(self, *outputs: Output, **kwargs) -> None:
        """Add an output to the writer.

        Args:
            *outputs: The outputs to add.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abc.abstractmethod
    def remove_output(self, output_name: str) -> None:
        """Remove an output from the writer.

        Args:
            output_name: The name of the output to remove.
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
