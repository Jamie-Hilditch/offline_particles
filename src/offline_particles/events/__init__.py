"""Submodule for working with simulation events."""

from ._events import Event, SimulationState
from ._schedulers import (
    AtIterationScheduler,
    AtTimeScheduler,
    IterationSchedulerProtocol,
    RecurringIterationScheduler,
    RecurringTimeScheduler,
    TimeSchedulerProtocol,
)

__all__ = [
    "AtIterationScheduler",
    "AtTimeScheduler",
    "Event",
    "IterationSchedulerProtocol",
    "RecurringIterationScheduler",
    "RecurringTimeScheduler",
    "SimulationState",
    "TimeSchedulerProtocol",
]

# Set __module__ for all public classes to this module for cleaner documentation
_module = __name__
for _cls in [
    AtIterationScheduler,
    AtTimeScheduler,
    Event,
    IterationSchedulerProtocol,
    RecurringIterationScheduler,
    RecurringTimeScheduler,
    SimulationState,
    TimeSchedulerProtocol,
]:
    _cls.__module__ = _module
