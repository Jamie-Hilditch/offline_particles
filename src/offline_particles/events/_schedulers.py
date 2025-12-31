"""Submodules for event schedulers."""

import dataclasses
from typing import Iterable

from ._events import Event


@dataclasses.dataclass(frozen=True)
class IterationSchedule:
    """Dataclass representing an iteration-based schedule."""

    N: int
    first: int = 0

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError("N must be a positive integer")
        if self.first < 0:
            raise ValueError("first must be a non-negative integer")


@dataclasses.dataclass(frozen=True)
class TimeSchedule:
    """Dataclass representing a time-based schedule."""

    dt: float
    first: float

    def validate(self, forward: bool, resolution: float) -> None:
        if forward and self.dt <= 0:
            raise ValueError("dt must be a positive number for forward time")
        if not forward and self.dt >= 0:
            raise ValueError("dt must be a negative number for backward time")
        if abs(self.dt) < resolution:
            raise ValueError(f"dt must be at least {resolution} in magnitude to avoid floating point issues")


type AbstractSchedule = IterationSchedule | TimeSchedule


class IterationScheduler:
    """A scheduler that triggers events every N iterations."""

    def __init__(self) -> None:
        self._next = None
        self._events: dict[int, list[tuple[int, Event]]] = dict()

    def _schedule_event(self, iteration: int, N: int, event: Event) -> None:
        if iteration not in self._events:
            self._events[iteration] = []
        self._events[iteration].append((N, event))

    @property
    def next(self) -> int | None:
        """The next iteration at which an event is scheduled."""
        return self._next

    @property
    def events(self) -> Iterable[Event]:
        """All registered events."""
        for event_list in self._events.values():
            for _, event in event_list:
                yield event

    def register_event(self, schedule: IterationSchedule, event: Event) -> None:
        """Register an event to be triggered every N iterations.

        Args:
            event (Event): The event to be triggered.
            schedule (IterationSchedule): The iteration schedule.
        """
        self._schedule_event(schedule.first, schedule.N, event)
        self.set_next()

    def set_next(self) -> None:
        """Set the next iteration to check for events."""
        self._next = min(self._events.keys()) if self._events else None

    def __call__(self, iteration: int) -> list[Event]:
        """Get the events to trigger at the given iteration.

        Args:
            iteration (int): The current iteration.

        Returns:
            list[Event]: The list of events to trigger.
        """
        triggered_events: list[Event] = []

        while self._next is not None and self._next <= iteration:
            for N, event in self._events.pop(self._next, []):
                triggered_events.append(event)
                # Reschedule the event for its next occurrence
                next_occurrence = self._next + N
                self._schedule_event(next_occurrence, N, event)

            self.set_next()
        return triggered_events


class TimeScheduler:
    """A scheduler that triggers events every dt."""

    def __init__(self, *, forward: bool = True, resolution: float = 1e-12) -> None:
        self._forward = forward
        self._next_tick = None
        self._resolution = resolution
        self._events: dict[int, list[tuple[float, Event]]] = dict()

    def _discretise_time(self, time: float) -> int:
        """Round time and use integer ticks to avoid floating point issues."""
        return round(time / self._resolution)

    def _time_from_tick(self, tick: int) -> float:
        return tick * self._resolution

    def _schedule_event(self, tick: int, dt: float, event: Event) -> None:
        if tick not in self._events:
            self._events[tick] = []
        self._events[tick].append((dt, event))

    @property
    def next_time(self) -> float | None:
        """The next time at which an event is scheduled."""
        if self._next_tick is None:
            return None
        return self._time_from_tick(self._next_tick)

    @property
    def events(self) -> Iterable[Event]:
        """All registered events."""
        for event_list in self._events.values():
            for _, event in event_list:
                yield event

    def register_event(self, schedule: TimeSchedule, event: Event) -> None:
        """Register an event to be triggered every dt.

        Args:
            event (Event): The event to be triggered.
            schedule (TimeSchedule): The time schedule.
        """
        schedule.validate(self._forward, self._resolution)
        first_tick = self._discretise_time(schedule.first)
        self._schedule_event(first_tick, schedule.dt, event)
        self.set_next()

    def set_next(self) -> None:
        """Set the next time to check for events."""
        if not self._events:
            self._next_tick = None
            return
        if self._forward:
            self._next_tick = min(self._events.keys())
        else:
            self._next_tick = max(self._events.keys())

    def __call__(self, time: float) -> list[Event]:
        """Get the events to trigger at the given time.

        Args:
            time (float): The current time.

        Returns:
            list[Event]: The list of events to trigger.
        """
        tick = self._discretise_time(time)
        triggered_events: list[Event] = []

        while (nt := self._next_tick) is not None and (nt <= tick if self._forward else nt >= tick):
            for dt, event in self._events.pop(nt, []):
                triggered_events.append(event)
                # Reschedule the event for its next occurrence
                next_occurrence = self._discretise_time(self._time_from_tick(nt) + dt)
                self._schedule_event(next_occurrence, dt, event)

            self.set_next()
        return triggered_events
