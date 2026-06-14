"""Tkinter-based window renderer for live captions (system audio only, no translation).

UiQueue is a thread-safe event bridge: ASR worker threads enqueue events, the Tk main
thread drains them via after() and applies them to widgets. UiQueue is intentionally
Tk-free so it can be unit-tested without a display.
"""

from __future__ import annotations

import queue
import threading
from datetime import datetime


class UiQueue:
    """Thread-safe FIFO bridge between ASR callbacks (worker thread) and Tk updates (main thread).

    Events are tuples:
      ("status", message: str)
      ("partial", text: str, started_at: datetime | None)
      ("final", text: str, started_at: datetime | None)
    """

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

    def enqueue_status(self, message: str) -> None:
        self._q.put(("status", message))

    def enqueue_partial(self, text: str, started_at: datetime | None) -> None:
        self._q.put(("partial", text, started_at))

    def enqueue_final(self, text: str, started_at: datetime | None) -> None:
        self._q.put(("final", text, started_at))

    def drain(self) -> list[tuple]:
        """Return all pending events in FIFO order, clearing the queue.

        Must be called from the Tk main thread only.
        """
        events: list[tuple] = []
        while True:
            try:
                events.append(self._q.get_nowait())
            except queue.Empty:
                break
        return events
