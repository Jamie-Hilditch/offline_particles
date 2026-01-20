"""Submodule for working with simulation events."""

import dataclasses
import types
from typing import Callable, Iterable, Mapping

import numpy as np

from ..kernels import ParticleKernel
from ..particles import ParticlesView


@dataclasses.dataclass(frozen=True)
class SimulationState:
    """Dataclass representing the current state of the simulation."""

    time: np.float64 | np.datetime64
    dt: np.float64 | np.timedelta64
    tidx: np.float64
    iteration: int
    wall_time: np.timedelta64
    particles: ParticlesView


type EventFunction = Callable[[SimulationState], None]


class Event:
    """A simulation event.

    An event consists of a single function that acts on the simulation state,
    along with any number of associated particle kernels that are launched by the
    scheduler prior to the invokation of the event function.

    Particle kernels are used to prepare or modify particle data before the event
    function is called. They are stored as a mapping from ParticleSet name to the kernel.
    """

    def __init__(self, name: str, func: EventFunction, **kernels: Iterable[ParticleKernel]) -> None:
        """Initialize the event."""
        self._name = name
        self._func = func
        self._kernels: dict[str, ParticleKernel] = {name: tuple(kernels) for name, kernels in kernels.items()}

    def __call__(self, state: SimulationState) -> None:
        """Invoke the event function."""
        self._func(state)

    @property
    def name(self) -> str:
        """The name of the event."""
        return self._name

    @property
    def kernels(self) -> Mapping[str, tuple[ParticleKernel, ...]]:
        """The kernels associated with this event."""
        return types.MappingProxyType(self._kernels)
