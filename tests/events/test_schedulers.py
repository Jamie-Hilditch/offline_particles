"""Tests for event schedulers."""

import numpy as np
import pytest

from offline_particles.events import (
    AtIterationScheduler,
    AtTimeScheduler,
    Event,
    IterationSchedulerProtocol,
    RecurringIterationScheduler,
    RecurringTimeScheduler,
    SimulationState,
    TimeSchedulerProtocol,
)


def _make_event(name: str = "test") -> Event:
    """Create a simple no-op event for testing.

    Parameters
    ----------
    name : str
        The name of the event. Defaults to "test".

    Returns
    -------
    Event
        An Event instance with the given name and a no-op function.
    """

    def noop(state: SimulationState) -> None:
        pass

    return Event(name, noop)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_recurring_iteration_scheduler_conforms_to_protocol(self) -> None:
        scheduler = RecurringIterationScheduler()
        assert isinstance(scheduler, IterationSchedulerProtocol)

    def test_at_iteration_scheduler_conforms_to_protocol(self) -> None:
        scheduler = AtIterationScheduler()
        assert isinstance(scheduler, IterationSchedulerProtocol)

    def test_recurring_time_scheduler_conforms_to_protocol(self) -> None:
        scheduler = RecurringTimeScheduler()
        assert isinstance(scheduler, TimeSchedulerProtocol)

    def test_at_time_scheduler_conforms_to_protocol(self) -> None:
        scheduler = AtTimeScheduler()
        assert isinstance(scheduler, TimeSchedulerProtocol)


# ---------------------------------------------------------------------------
# RecurringIterationScheduler
# ---------------------------------------------------------------------------


class TestRecurringIterationScheduler:
    def test_initially_empty(self) -> None:
        scheduler = RecurringIterationScheduler()
        assert scheduler.next is None
        assert list(scheduler.events) == []

    def test_register_event_sets_next(self) -> None:
        scheduler = RecurringIterationScheduler()
        event = _make_event()
        scheduler.register_event(5, 10, event)
        assert scheduler.next == 5

    def test_triggers_at_first_iteration(self) -> None:
        scheduler = RecurringIterationScheduler()
        event = _make_event()
        scheduler.register_event(0, 5, event)
        result = scheduler(0)
        assert result == [event]

    def test_does_not_trigger_before_first(self) -> None:
        scheduler = RecurringIterationScheduler()
        event = _make_event()
        scheduler.register_event(5, 5, event)
        assert scheduler(4) == []

    def test_recurring_trigger(self) -> None:
        scheduler = RecurringIterationScheduler()
        event = _make_event()
        scheduler.register_event(0, 3, event)
        assert scheduler(0) == [event]
        assert scheduler(3) == [event]
        assert scheduler(6) == [event]

    def test_no_duplicate_trigger_between_intervals(self) -> None:
        scheduler = RecurringIterationScheduler()
        event = _make_event()
        scheduler.register_event(0, 5, event)
        scheduler(0)
        assert scheduler(4) == []

    def test_multiple_events_at_same_iteration(self) -> None:
        scheduler = RecurringIterationScheduler()
        e1 = _make_event("e1")
        e2 = _make_event("e2")
        scheduler.register_event(0, 5, e1)
        scheduler.register_event(0, 5, e2)
        result = scheduler(0)
        assert set(result) == {e1, e2}

    def test_events_property_lists_all_registered(self) -> None:
        scheduler = RecurringIterationScheduler()
        e1 = _make_event("e1")
        e2 = _make_event("e2")
        scheduler.register_event(0, 5, e1)
        scheduler.register_event(10, 5, e2)
        assert set(scheduler.events) == {e1, e2}


# ---------------------------------------------------------------------------
# AtIterationScheduler
# ---------------------------------------------------------------------------


class TestAtIterationScheduler:
    def test_initially_empty(self) -> None:
        scheduler = AtIterationScheduler()
        assert scheduler.next is None
        assert list(scheduler.events) == []

    def test_register_event_sets_next(self) -> None:
        scheduler = AtIterationScheduler()
        event = _make_event()
        scheduler.register_event(5, event)
        assert scheduler.next == 5

    def test_triggers_at_scheduled_iteration(self) -> None:
        scheduler = AtIterationScheduler()
        event = _make_event()
        scheduler.register_event(3, event)
        result = scheduler(3)
        assert result == [event]

    def test_does_not_trigger_before_scheduled_iteration(self) -> None:
        scheduler = AtIterationScheduler()
        event = _make_event()
        scheduler.register_event(5, event)
        assert scheduler(4) == []

    def test_fires_only_once(self) -> None:
        scheduler = AtIterationScheduler()
        event = _make_event()
        scheduler.register_event(3, event)
        # First call at the scheduled iteration
        assert scheduler(3) == [event]
        # Should not fire again
        assert scheduler(4) == []
        assert scheduler(10) == []

    def test_fires_at_current_iteration_when_past_due(self) -> None:
        """An event scheduled for iteration 3 should fire when called at iteration 5."""
        scheduler = AtIterationScheduler()
        event = _make_event()
        scheduler.register_event(3, event)
        result = scheduler(5)
        assert result == [event]

    def test_multiple_events_at_same_iteration(self) -> None:
        scheduler = AtIterationScheduler()
        e1 = _make_event("e1")
        e2 = _make_event("e2")
        scheduler.register_event(5, e1)
        scheduler.register_event(5, e2)
        result = scheduler(5)
        assert set(result) == {e1, e2}

    def test_multiple_events_at_different_iterations(self) -> None:
        scheduler = AtIterationScheduler()
        e1 = _make_event("e1")
        e2 = _make_event("e2")
        scheduler.register_event(2, e1)
        scheduler.register_event(5, e2)
        assert scheduler(2) == [e1]
        assert scheduler(5) == [e2]

    def test_empty_after_all_events_fired(self) -> None:
        scheduler = AtIterationScheduler()
        event = _make_event()
        scheduler.register_event(3, event)
        scheduler(3)
        assert scheduler.next is None
        assert list(scheduler.events) == []

    def test_events_property_lists_all_registered(self) -> None:
        scheduler = AtIterationScheduler()
        e1 = _make_event("e1")
        e2 = _make_event("e2")
        scheduler.register_event(2, e1)
        scheduler.register_event(5, e2)
        assert set(scheduler.events) == {e1, e2}


# ---------------------------------------------------------------------------
# RecurringTimeScheduler
# ---------------------------------------------------------------------------


class TestRecurringTimeScheduler:
    def test_initially_empty(self) -> None:
        scheduler = RecurringTimeScheduler()
        assert scheduler.next_time is None
        assert list(scheduler.events) == []

    def test_register_event_sets_next_time(self) -> None:
        scheduler = RecurringTimeScheduler()
        event = _make_event()
        scheduler.register_event(np.float64(0.0), np.float64(1.0), event)
        assert scheduler.next_time == np.float64(0.0)

    def test_triggers_at_first_time(self) -> None:
        scheduler = RecurringTimeScheduler()
        event = _make_event()
        scheduler.register_event(np.float64(0.0), np.float64(1.0), event)
        result = scheduler(np.float64(0.0))
        assert result == [event]

    def test_does_not_trigger_before_first_time(self) -> None:
        scheduler = RecurringTimeScheduler()
        event = _make_event()
        scheduler.register_event(np.float64(1.0), np.float64(1.0), event)
        assert scheduler(np.float64(0.5)) == []

    def test_recurring_trigger(self) -> None:
        scheduler = RecurringTimeScheduler()
        event = _make_event()
        scheduler.register_event(np.float64(0.0), np.float64(1.0), event)
        assert scheduler(np.float64(0.0)) == [event]
        assert scheduler(np.float64(1.0)) == [event]
        assert scheduler(np.float64(2.0)) == [event]

    def test_rejects_non_positive_dt_when_forward(self) -> None:
        scheduler = RecurringTimeScheduler(forward_in_time=True)
        event = _make_event()
        with pytest.raises(ValueError, match="dt must be positive"):
            scheduler.register_event(np.float64(0.0), np.float64(-1.0), event)

    def test_rejects_non_negative_dt_when_backward(self) -> None:
        scheduler = RecurringTimeScheduler(forward_in_time=False)
        event = _make_event()
        with pytest.raises(ValueError, match="dt must be negative"):
            scheduler.register_event(np.float64(0.0), np.float64(1.0), event)

    def test_backward_in_time(self) -> None:
        scheduler = RecurringTimeScheduler(forward_in_time=False)
        event = _make_event()
        scheduler.register_event(np.float64(10.0), np.float64(-1.0), event)
        assert scheduler(np.float64(10.0)) == [event]
        assert scheduler(np.float64(9.0)) == [event]


# ---------------------------------------------------------------------------
# AtTimeScheduler
# ---------------------------------------------------------------------------


class TestAtTimeScheduler:
    def test_initially_empty(self) -> None:
        scheduler = AtTimeScheduler()
        assert scheduler.next_time is None
        assert list(scheduler.events) == []

    def test_register_event_sets_next_time(self) -> None:
        scheduler = AtTimeScheduler()
        event = _make_event()
        scheduler.register_event(np.float64(5.0), event)
        assert scheduler.next_time == np.float64(5.0)

    def test_triggers_at_scheduled_time(self) -> None:
        scheduler = AtTimeScheduler()
        event = _make_event()
        scheduler.register_event(np.float64(3.0), event)
        result = scheduler(np.float64(3.0))
        assert result == [event]

    def test_does_not_trigger_before_scheduled_time(self) -> None:
        scheduler = AtTimeScheduler()
        event = _make_event()
        scheduler.register_event(np.float64(5.0), event)
        assert scheduler(np.float64(4.0)) == []

    def test_fires_only_once(self) -> None:
        scheduler = AtTimeScheduler()
        event = _make_event()
        scheduler.register_event(np.float64(3.0), event)
        assert scheduler(np.float64(3.0)) == [event]
        # Should not fire again
        assert scheduler(np.float64(4.0)) == []
        assert scheduler(np.float64(10.0)) == []

    def test_fires_when_past_due(self) -> None:
        """An event scheduled for time 3.0 should fire when called at time 5.0."""
        scheduler = AtTimeScheduler()
        event = _make_event()
        scheduler.register_event(np.float64(3.0), event)
        result = scheduler(np.float64(5.0))
        assert result == [event]

    def test_multiple_events_at_same_time(self) -> None:
        scheduler = AtTimeScheduler()
        e1 = _make_event("e1")
        e2 = _make_event("e2")
        scheduler.register_event(np.float64(5.0), e1)
        scheduler.register_event(np.float64(5.0), e2)
        result = scheduler(np.float64(5.0))
        assert set(result) == {e1, e2}

    def test_multiple_events_at_different_times(self) -> None:
        scheduler = AtTimeScheduler()
        e1 = _make_event("e1")
        e2 = _make_event("e2")
        scheduler.register_event(np.float64(2.0), e1)
        scheduler.register_event(np.float64(5.0), e2)
        assert scheduler(np.float64(2.0)) == [e1]
        assert scheduler(np.float64(5.0)) == [e2]

    def test_empty_after_all_events_fired(self) -> None:
        scheduler = AtTimeScheduler()
        event = _make_event()
        scheduler.register_event(np.float64(3.0), event)
        scheduler(np.float64(3.0))
        assert scheduler.next_time is None
        assert list(scheduler.events) == []

    def test_backward_in_time_triggers_correctly(self) -> None:
        scheduler = AtTimeScheduler(forward_in_time=False)
        event = _make_event()
        scheduler.register_event(np.float64(5.0), event)
        # Should trigger at 5.0 or below
        assert scheduler(np.float64(5.0)) == [event]

    def test_backward_in_time_does_not_trigger_before_scheduled(self) -> None:
        scheduler = AtTimeScheduler(forward_in_time=False)
        event = _make_event()
        scheduler.register_event(np.float64(5.0), event)
        # When going backward, 6.0 is "before" 5.0
        assert scheduler(np.float64(6.0)) == []

    def test_backward_in_time_fires_only_once(self) -> None:
        scheduler = AtTimeScheduler(forward_in_time=False)
        event = _make_event()
        scheduler.register_event(np.float64(5.0), event)
        assert scheduler(np.float64(5.0)) == [event]
        assert scheduler(np.float64(4.0)) == []
