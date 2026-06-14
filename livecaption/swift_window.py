"""Swift-based native macOS caption window.

Spawns the livecaption-window binary as a subprocess and sends caption
events as JSON Lines via stdin. Mirrors the public interface of the old
WindowRenderer so cli_window.py needs minimal changes.
"""

from __future__ import annotations

import contextlib
import json
import queue
import subprocess
import threading
from datetime import datetime


class SwiftCaptionWindow:
    """Native Swift macOS caption window — replacement for tkinter WindowRenderer.

    ASR callbacks (from the worker thread) enqueue JSON objects; a dedicated
    writer thread serialises them to the Swift subprocess's stdin.

    Lifecycle:
        window = SwiftCaptionWindow(binary_path)
        window.show()            # blocks until the window closes
        # or:
        window.close()           # called from another thread to request shutdown
    """

    def __init__(self, binary_path: str) -> None:
        self._binary = binary_path
        self._proc: subprocess.Popen[str] | None = None
        self._stop_event = threading.Event()
        self._q: queue.Queue[str] = queue.Queue()
        self._writer: threading.Thread | None = None

    # ---- Public API (callable from any thread) ----

    def set_stop_event(self, event: threading.Event) -> None:
        """Share the CLI-level stop_event so window-close signals propagate out."""
        self._stop_event = event

    def partial(
        self, label: str, text: str, started_at: datetime, speaker: int | None = None
    ) -> None:
        """Called from AsrWorker thread. Send live partial to the Swift window."""
        self._send({"type": "partial", "text": text})

    def final(self, label: str, segments: list, started_at: datetime) -> None:
        """Called from AsrWorker thread. Send finalized transcript to the Swift window.

        segments = [(speaker, text, diff), ...]; flattened to plain text
        (the window does not show speaker labels or diff spans).
        """
        parts = [seg[1] for seg in segments]
        text = "  ".join(parts)
        self._send({"type": "final", "text": text})

    def set_status(self, message: str) -> None:
        """Update the status bar (thread-safe)."""
        self._send({"type": "status", "message": message})

    def show(self) -> None:
        """Spawn the Swift window and block until it exits."""
        self._proc = subprocess.Popen(
            [self._binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._writer = threading.Thread(
            target=self._write_loop, daemon=True, name="swift-window-writer"
        )
        self._writer.start()
        with contextlib.suppress(Exception):
            self._proc.wait()

    def close(self) -> None:
        """Request shutdown from another thread (signal handler or error callback)."""
        self._stop_event.set()
        if self._proc is not None:
            with contextlib.suppress(Exception):
                self._proc.terminate()
                self._proc.wait(timeout=2)
            if self._proc.poll() is None:
                with contextlib.suppress(Exception):
                    self._proc.kill()

    # ---- Internal ----

    def _send(self, obj: dict) -> None:
        """Enqueue a JSON-line event (thread-safe)."""
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self._q.put(line)

    def _write_loop(self) -> None:
        """Drain the internal queue and write JSON lines to the subprocess stdin."""
        while not self._stop_event.is_set():
            try:
                line = self._q.get(timeout=0.2)
            except queue.Empty:
                # Check if the subprocess died — if so, stop writing
                if self._proc is not None and self._proc.poll() is not None:
                    break
                continue
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.stdin.write(line)
                    self._proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    break
