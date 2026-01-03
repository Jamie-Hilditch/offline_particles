"""Submodule for event schedulers."""

from typing import Iterable

import numpy as np

from ._events import Event

type T = np.float64 | np.datetime64
type D = np.float64 | np.timedelta64


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

    def register_event(self, first: int, n: int, event: Event) -> None:
        """Register an event to be triggered every N iterations.

        Args:
            first (int): The first iteration the event is triggered.
            n (int): The number of iterations between events.
            event (Event): The event to be triggered.
        """
        self._schedule_event(first, n, event)
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

    def __init__(self, *, forward: bool = True) -> None:
        self._forward = forward

        self._next_time = None
        self._events: dict[T, list[tuple[D, Event]]] = dict()

    def _schedule_event(self, time: T, dt: D, event: Event) -> None:
        if time not in self._events:
            self._events[time] = []
        self._events[time].append((dt, event))

    @property
    def next_time(self) -> T | None:
        """The next time at which an event is scheduled."""
        return self._next_time

    @property
    def events(self) -> Iterable[Event]:
        """All registered events."""
        for event_list in self._events.values():
            for _, event in event_list:
                yield event

    def register_event(self, first: T, dt: D, event: Event) -> None:
        """Register an event to be triggered every dt.

        Args:
            first (T): The first time the event is triggered.
            dt (D): The time interval between events.
            event (Event): The event to be triggered.
        """
        self._schedule_event(first, dt, event)
        self.set_next()

    def set_next(self) -> None:
        """Set the next time to check for events."""
        if not self._events:
            self._next_time = None
            return
        if self._forward:
            self._next_time = min(self._events.keys())
        else:
            self._next_time = max(self._events.keys())

    def __call__(self, time: T) -> list[Event]:
        """Get the events to trigger at the given time.

        Args:
            time (float): The current time.

        Returns:
            list[Event]: The list of events to trigger.
        """
        triggered_events: list[Event] = []

        while (nt := self._next_time) is not None and (nt <= time if self._forward else nt >= time):
            for dt, event in self._events.pop(nt, []):
                triggered_events.append(event)
                # Reschedule the event for its next occurrence
                next_occurrence = nt + dt  # type: ignore[operator]
                self._schedule_event(next_occurrence, dt, event)

            self.set_next()
        return triggered_events
