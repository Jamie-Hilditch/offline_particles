"""Submodule for event schedulers."""

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from ..timestepping import D, T
from ._events import Event


@runtime_checkable
class IterationSchedulerProtocol(Protocol):
    """Protocol for iteration-based event schedulers."""

    def events(self) -> Iterable[Event]:
        """Yield all registered events.

        Yields
        ------
        event : Event
            The next registered event.
        """
        ...

    def __call__(self, iteration: int) -> list[Event]:
        """Get the events to trigger at the given iteration."""
        ...


@runtime_checkable
class TimeSchedulerProtocol(Protocol):
    """Protocol for time-based event schedulers."""

    def events(self) -> Iterable[Event]:
        """Yield all registered events.

        Yields
        ------
        event : Event
            The next registered event.
        """
        ...

    def __call__(self, time: T) -> list[Event]:
        """Get the events to trigger at the given time."""
        ...


class RecurringIterationScheduler:
    """A scheduler that triggers events every N iterations."""

    def __init__(self) -> None:
        self._next = None
        self._events: dict[int, list[tuple[int, Event]]] = {}

    def _schedule_event(self, iteration: int, N: int, event: Event) -> None:
        if iteration not in self._events:
            self._events[iteration] = []
        self._events[iteration].append((N, event))

    @property
    def next(self) -> int | None:
        """The next iteration at which an event is scheduled."""
        return self._next

    def events(self) -> Iterable[Event]:
        """Yield all registered events.

        Yields
        ------
        event : Event
            The next registered event.
        """
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

        Parameters
        ----------
        iteration : int
            The current iteration.

        Returns
        -------
        list[Event]
            The list of events to trigger.
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


class RecurringTimeScheduler:
    """A scheduler that triggers events every dt."""

    def __init__(self, *, forward_in_time: bool = True) -> None:
        self._forward_in_time = forward_in_time

        self._next_time = None
        self._events: dict[T, list[tuple[D, Event]]] = {}

    def _schedule_event(self, time: T, dt: D, event: Event) -> None:
        if time not in self._events:
            self._events[time] = []
        self._events[time].append((dt, event))

    @property
    def next_time(self) -> T | None:
        """The next time at which an event is scheduled."""
        return self._next_time

    def events(self) -> Iterable[Event]:
        """Yield all registered events.

        Yields
        ------
        event : Event
            The next registered event.
        """
        for event_list in self._events.values():
            for _, event in event_list:
                yield event

    def register_event(self, first: T, dt: D, event: Event) -> None:
        """Register an event to be triggered every dt.

        Parameters
        ----------
        first : T
            The first time the event is triggered.
        dt : D
            The time interval between events.
        event : Event
            The event to be triggered.

        Raises
        ------
        ValueError
            If dt is not positive when forward_in_time is True,
            or if dt is not negative when forward_in_time is False.
        """
        # validate dt
        if self._forward_in_time and not (dt > dt * 0):
            raise ValueError("dt must be positive when forward_in_time is True.")
        if not self._forward_in_time and not (dt < dt * 0):
            raise ValueError("dt must be negative when forward_in_time is False.")
        self._schedule_event(first, dt, event)
        self.set_next()

    def set_next(self) -> None:
        """Set the next time to check for events."""
        if not self._events:
            self._next_time = None
            return
        if self._forward_in_time:
            self._next_time = min(self._events.keys())
        else:
            self._next_time = max(self._events.keys())

    def __call__(self, time: T) -> list[Event]:
        """Get the events to trigger at the given time.

        Parameters
        ----------
        time : T
            The current time.

        Returns
        -------
        list[Event]
            The list of events to trigger.
        """
        triggered_events: list[Event] = []

        while True:
            nt = self._next_time
            # break conditions
            if nt is None:
                break
            if self._forward_in_time and nt > time:
                break
            if not self._forward_in_time and nt < time:
                break
            # trigger events and reschedule
            for dt, event in self._events.pop(nt, []):
                triggered_events.append(event)
                # Reschedule the event for its next occurrence
                next_occurrence = nt + dt  # type: ignore[operator]
                self._schedule_event(next_occurrence, dt, event)

            self.set_next()
        return triggered_events


class AtIterationScheduler:
    """A scheduler that triggers events once at a specific iteration."""

    def __init__(self) -> None:
        self._next = None
        self._events: dict[int, list[Event]] = {}

    def _schedule_event(self, iteration: int, event: Event) -> None:
        if iteration not in self._events:
            self._events[iteration] = []
        self._events[iteration].append(event)

    @property
    def next(self) -> int | None:
        """The next iteration at which an event is scheduled."""
        return self._next

    def events(self) -> Iterable[Event]:
        """Yield all registered events.

        Yields
        ------
        event : Event
            The next registered event.
        """
        for event_list in self._events.values():
            yield from event_list

    def register_event(self, iteration: int, event: Event) -> None:
        """Register an event to be triggered once at the given iteration.

        Parameters
        ----------
        iteration : int
            The iteration at which to trigger the event.
        event : Event
            The event to be triggered.
        """
        self._schedule_event(iteration, event)
        self.set_next()

    def set_next(self) -> None:
        """Set the next iteration to check for events."""
        self._next = min(self._events.keys()) if self._events else None

    def __call__(self, iteration: int) -> list[Event]:
        """Get the events to trigger at the given iteration.

        Parameters
        ----------
        iteration : int
            The current iteration.

        Returns
        -------
        list[Event]
            The list of events to trigger.
        """
        triggered_events: list[Event] = []

        while self._next is not None and self._next <= iteration:
            # Events are not rescheduled — they fire once and are removed
            triggered_events.extend(self._events.pop(self._next, []))
            self.set_next()

        return triggered_events


class AtTimeScheduler:
    """A scheduler that triggers events once at a specific time."""

    def __init__(self, *, forward_in_time: bool = True) -> None:
        self._forward_in_time = forward_in_time
        self._next_time = None
        self._events: dict[T, list[Event]] = {}

    def _schedule_event(self, time: T, event: Event) -> None:
        if time not in self._events:
            self._events[time] = []
        self._events[time].append(event)

    @property
    def next_time(self) -> T | None:
        """The next time at which an event is scheduled."""
        return self._next_time

    def events(self) -> Iterable[Event]:
        """Yield all registered events.

        Yields
        ------
        event : Event
            The next registered event.
        """
        for event_list in self._events.values():
            yield from event_list

    def register_event(self, time: T, event: Event) -> None:
        """Register an event to be triggered once at the given time.

        Parameters
        ----------
        time : T
            The time at which to trigger the event.
        event : Event
            The event to be triggered.
        """
        self._schedule_event(time, event)
        self.set_next()

    def set_next(self) -> None:
        """Set the next time to check for events."""
        if not self._events:
            self._next_time = None
            return
        if self._forward_in_time:
            self._next_time = min(self._events.keys())
        else:
            self._next_time = max(self._events.keys())

    def __call__(self, time: T) -> list[Event]:
        """Get the events to trigger at the given time.

        Parameters
        ----------
        time : T
            The current time.

        Returns
        -------
        list[Event]
            The list of events to trigger.
        """
        triggered_events: list[Event] = []

        while True:
            nt = self._next_time
            # break conditions
            if nt is None:
                break
            if self._forward_in_time and nt > time:
                break
            if not self._forward_in_time and nt < time:
                break
            # Events are not rescheduled — they fire once and are removed
            triggered_events.extend(self._events.pop(nt, []))
            self.set_next()

        return triggered_events
