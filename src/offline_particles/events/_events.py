"""Submodule for working with simulation events."""

import dataclasses
from typing import Callable

from ..kernels import ParticleKernel
from ..particles import ParticlesView


@dataclasses.dataclass(frozen=True)
class SimulationState:
    """Dataclass representing the current state of the simulation."""

    time: float
    dt: float
    tidx: float
    particles: ParticlesView


type EventFunction = Callable[[SimulationState], None]


class Event:
    """A simulation event.

    An event consists of a single function that acts on the simulation state,
    along with any number of associated particle kernels that are launched by the
    scheduler prior to the invokation of the event function.
    """

    def __init__(self, name: str, func: EventFunction, *kernels: ParticleKernel) -> None:
        """Initialize the event."""
        self._name = name
        self._func = func
        self._kernels = kernels

    def __call__(self, state: SimulationState) -> None:
        """Invoke the event function."""
        self._func(state)

    @property
    def name(self) -> str:
        """The name of the event."""
        return self._name

    def kernels(self) -> tuple[ParticleKernel, ...]:
        """The kernels associated with this event."""
        return self._kernels
