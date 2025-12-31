"""Submodule for declaring output."""

import abc
import dataclasses
import functools
from typing import Mapping

import numpy as np

from ..events import AbstractSchedule, Event, SimulationState
from ..kernels import ParticleKernel, merge_particle_fields


@dataclasses.dataclass(frozen=True)
class Output:
    """Class defining a single output."""

    name: str
    particle_field: str
    kernels: tuple[ParticleKernel, ...]
    dtype: np.dtype = dataclasses.field(init=False)

    def __init__(
        self,
        name: str,
        particle_field: str,
        *kernels: ParticleKernel,
    ) -> None:
        """Initialize the Output."""
        kernel_fields = merge_particle_fields(kernels)
        if particle_field not in kernel_fields:
            raise ValueError(f"Particle field '{particle_field}' not found in provided kernels.")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "particle_field", particle_field)
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "dtype", kernel_fields[particle_field])


class AbstractOutputWriter(abc.ABC):
    """Interface for output writers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The name of the output writer."""
        pass

    @property
    @abc.abstractmethod
    def schedule(self) -> AbstractSchedule:
        """The output schedule."""
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
            name: The name of the output variable to write.
            particles: The current view of the particles.
        """
        pass

    @abc.abstractmethod
    def finalise_write_round(self, state: SimulationState) -> None:
        """Confirm that all outputs have been written for the current round."""
        pass

    def event_name(self, output_name: str) -> str:
        """Generate a name for an output event.

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
            event = Event(self.event_name(name), event_func, *output.kernels)
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
    def schedule(self) -> AbstractSchedule:
        """The output schedule."""
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
    def remove_output(self, name: str) -> None:
        """Remove an output from the writer.

        Args:
            name: The name of the output to remove.
        """
        pass

    @abc.abstractmethod
    def build(
        self,
        nparticles: int,
    ) -> AbstractOutputWriter:
        """Build the output writer."""
        pass
