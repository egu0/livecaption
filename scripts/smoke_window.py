"""Smoke tests for the window renderer: UiQueue marshaling + entry-point construction defaults."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

# Allow running from repo root: uv run python scripts/smoke_window.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livecaption.window import UiQueue


def test_enqueue_and_drain_order():
    """Events drain in FIFO order, exactly as enqueued."""
    q = UiQueue()
    q.enqueue_status("loading")
    q.enqueue_partial("hello", None)
    q.enqueue_status("listening")
    q.enqueue_final("hello world", None)
    q.enqueue_partial("next", None)

    events = q.drain()
    assert events == [
        ("status", "loading"),
        ("partial", "hello", None),
        ("status", "listening"),
        ("final", "hello world", None),
        ("partial", "next", None),
    ], f"Unexpected events: {events}"


def test_drain_clears_queue():
    """After drain, the queue is empty."""
    q = UiQueue()
    q.enqueue_partial("a", None)
    q.drain()
    assert q.drain() == []


def test_thread_safety():
    """Concurrent enqueues from multiple threads preserve order within each thread,
    and all events are eventually drained."""
    q = UiQueue()
    n = 50
    barrier = threading.Barrier(2)

    def producer(prefix: str) -> None:
        barrier.wait()
        for i in range(n):
            q.enqueue_partial(f"{prefix}-{i}", None)

    t1 = threading.Thread(target=producer, args=("A",))
    t2 = threading.Thread(target=producer, args=("B",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # All 2*n events should be drainable (order is non-deterministic between threads,
    # so we just check count and that no event is lost).
    events = q.drain()
    assert len(events) == 2 * n, f"Expected {2 * n} events, got {len(events)}"
    # Each producer's events should appear in order within the full drain list
    a_events = [e for e in events if e[1].startswith("A-")]
    b_events = [e for e in events if e[1].startswith("B-")]
    assert len(a_events) == n
    assert len(b_events) == n
    for i, ev in enumerate(a_events):
        assert ev[1] == f"A-{i}"
    for i, ev in enumerate(b_events):
        assert ev[1] == f"B-{i}"


def test_status_event():
    """Status events have the correct shape."""
    q = UiQueue()
    q.enqueue_status("error: audiotee not found")
    events = q.drain()
    assert events == [("status", "error: audiotee not found")]


def test_partial_event_with_timestamp():
    """Partial events carry text and a timestamp placeholder."""
    from datetime import datetime
    ts = datetime.now()
    q = UiQueue()
    q.enqueue_partial("testing partial", ts)
    events = q.drain()
    assert events[0][0] == "partial"
    assert events[0][1] == "testing partial"
    assert events[0][2] is ts


if __name__ == "__main__":
    # Simple test runner — no framework dependency
    tests = [
        test_enqueue_and_drain_order,
        test_drain_clears_queue,
        test_thread_safety,
        test_status_event,
        test_partial_event_with_timestamp,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    if passed < len(tests):
        sys.exit(1)
