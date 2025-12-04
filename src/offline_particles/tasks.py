"""Submodule for handling simulation tasks."""

import abc
import dataclasses
from typing import Iterable

import numpy as np

from .kernels import ParticleKernel
from .launcher import Launcher


@dataclasses.dataclass(frozen=True)
class SimulationState:
    """Dataclass representing the current state of the simulation."""

    time: float
    dt: float
    tidx: float
    particles: np.ndarray


class Task(abc.ABC):
    """Base class for simulation tasks."""

    @abc.abstractmethod
    def kernels(self) -> Iterable[ParticleKernel]:
        """Return the kernels associated with this task.

        Returns:
            An iterable of kernels run during this task.
        """
        pass

    @abc.abstractmethod
    def should_execute(self, state: SimulationState) -> bool:
        """Determine if the task should execute at the current simulation state.

        Args:
            state (SimulationState): The current state of the simulation.
        Returns:
            bool: True if the task should execute, False otherwise.
        """
        pass

    @abc.abstractmethod
    def execute(self, state: SimulationState, launcher: Launcher) -> None:
        """Execute the task's operations."""
        pass
