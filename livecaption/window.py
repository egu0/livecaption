"""Tkinter-based window renderer for live captions (system audio only, no translation).

UiQueue is a thread-safe event bridge: ASR worker threads enqueue events, the Tk main
thread drains them via after() and applies them to widgets. UiQueue is intentionally
Tk-free so it can be unit-tested without a display.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import ttk


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


# How often (ms) the Tk event loop drains the UiQueue
_DRAIN_INTERVAL_MS = 50

# Window geometry defaults
_WIN_WIDTH = 800
_WIN_HEIGHT = 500

# Font sizes (points)
_FONT_STATUS = 13
_FONT_TRANSCRIPT = 16
_FONT_PARTIAL = 16

# Partial widget: fixed height in text lines
_PARTIAL_LINES = 3

# Transcript history widget: keep at most this many lines to bound memory
_MAX_HISTORY_LINES = 500


class WindowRenderer:
    """Tkinter window for real-time caption display (system audio only, no translation).

    Owns the Tk root window and all widgets. ASR callbacks (from the worker thread)
    enqueue events into the UiQueue; the Tk main loop drains them via after() and
    applies updates to widgets on the main thread.

    Lifecycle:
        renderer = WindowRenderer()
        renderer.show()            # blocks until the window closes
        # or:
        renderer.close()           # called from another thread to request shutdown
    """

    def __init__(self) -> None:
        self._ui = UiQueue()
        self._stop_event = threading.Event()

        # ---- Tk root ----
        self._root = tk.Tk()
        self._root.title("LiveCaption")
        self._root.geometry(f"{_WIN_WIDTH}x{_WIN_HEIGHT}")
        self._root.configure(bg="#ffffff")
        # Minimize flash: pull the window onto the screen early
        self._root.update_idletasks()

        # ---- Status bar (top) ----
        status_font = ("TkDefaultFont", _FONT_STATUS)
        self._status_var = tk.StringVar(value="Starting…")
        self._status_label = ttk.Label(
            self._root, textvariable=self._status_var, font=status_font,
            padding=(12, 6), foreground="#555555",
        )
        self._status_label.pack(fill=tk.X, side=tk.TOP)

        ttk.Separator(self._root, orient=tk.HORIZONTAL).pack(fill=tk.X, side=tk.TOP)

        # ---- Transcript history area (middle, scrollable) ----
        transcript_font = ("TkDefaultFont", _FONT_TRANSCRIPT)
        self._transcript_text = tk.Text(
            self._root,
            font=transcript_font,
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            borderwidth=0,
            padx=12,
            pady=6,
            bg="#ffffff",
            fg="#1a1a1a",
            selectbackground="#c0c0c0",
        )
        scrollbar = ttk.Scrollbar(self._root, command=self._transcript_text.yview)
        self._transcript_text.configure(yscrollcommand=scrollbar.set)

        self._transcript_text.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)

        ttk.Separator(self._root, orient=tk.HORIZONTAL).pack(fill=tk.X, side=tk.TOP)

        # ---- Current partial area (bottom, fixed height) ----
        partial_font = ("TkDefaultFont", _FONT_PARTIAL)
        self._partial_text = tk.Text(
            self._root,
            font=partial_font,
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            borderwidth=0,
            padx=12,
            pady=10,
            height=_PARTIAL_LINES,
            bg="#fafafa",
            fg="#888888",
            selectbackground="#c0c0c0",
        )
        self._partial_text.pack(fill=tk.X, side=tk.BOTTOM)

        # ---- Window close → stop ----
        self._root.protocol("WM_DELETE_WINDOW", self._on_window_close)

    # ---- Public API (callable from any thread) ----

    def set_stop_event(self, event: threading.Event) -> None:
        """Share the CLI-level stop_event so window-close signals propagate out."""
        self._stop_event = event

    def partial(
        self, label: str, text: str, started_at: datetime, speaker: int | None = None
    ) -> None:
        """Called from AsrWorker thread. Enqueue partial update.

        Note: speaker is accepted to match the AsrWorker.on_partial callback contract
        but is ignored (the window does not show speaker labels).
        """
        self._ui.enqueue_partial(text, started_at)

    def final(self, label: str, segments: list, started_at: datetime) -> None:
        """Called from AsrWorker thread. Enqueue finalized transcript.

        segments = [(speaker, text, diff), ...]; without diarization, speaker is always None.
        We flatten to plain text (ignoring diff spans — the window always shows clean text).
        """
        parts = [seg[1] for seg in segments]
        text = "  ".join(parts)
        self._ui.enqueue_final(text, started_at)

    def set_status(self, message: str) -> None:
        """Update the status bar (thread-safe)."""
        self._ui.enqueue_status(message)

    def show(self) -> None:
        """Enter the Tk main loop. Blocks until the window is closed."""
        self._root.after(_DRAIN_INTERVAL_MS, self._drain_queue)
        self._root.mainloop()

    def close(self) -> None:
        """Request shutdown from another thread (e.g. signal handler or error callback)."""
        self._root.after(0, self._root.quit)

    # ---- Internal ----

    def _on_window_close(self) -> None:
        """User clicked the window close button."""
        self._stop_event.set()
        self._root.quit()

    def _drain_queue(self) -> None:
        """Pull pending events from UiQueue and apply to widgets.

        Scheduled via after() so it runs on the Tk main thread.
        Exits early if the window is being destroyed.
        """
        try:
            # If the window was destroyed, stop draining
            if not self._root.winfo_exists():  # type: ignore[no-untyped-call]
                return
        except tk.TclError:
            return

        for event in self._ui.drain():
            kind = event[0]
            if kind == "status":
                self._apply_status(event[1])
            elif kind == "partial":
                self._apply_partial(event[1], event[2])
            elif kind == "final":
                self._apply_final(event[1], event[2])

        with contextlib.suppress(tk.TclError):
            self._root.after(_DRAIN_INTERVAL_MS, self._drain_queue)

    def _apply_status(self, message: str) -> None:
        self._status_var.set(message)
        # Color-code status: green for listening, red for errors
        if message.lower().startswith("error") or message.lower().startswith("fatal"):
            self._status_label.configure(foreground="#cc0000")
        elif "listening" in message.lower():
            self._status_label.configure(foreground="#008800")
        else:
            self._status_label.configure(foreground="#555555")

    def _apply_partial(self, text: str, started_at: datetime | None) -> None:
        ts = f"[{started_at:%H:%M:%S}] " if started_at else ""
        self._partial_text.configure(state=tk.NORMAL)
        self._partial_text.delete("1.0", tk.END)
        self._partial_text.insert("1.0", f"{ts}{text}")
        self._partial_text.configure(state=tk.DISABLED)

    def _apply_final(self, text: str, started_at: datetime | None) -> None:
        ts = f"[{started_at:%H:%M:%S}] " if started_at else ""
        self._transcript_text.configure(state=tk.NORMAL)
        self._transcript_text.insert(tk.END, f"{ts}{text}\n")
        # Bound history lines
        line_count = int(self._transcript_text.index("end-1c").split(".")[0])
        if line_count > _MAX_HISTORY_LINES:
            # Delete oldest lines to stay under the cap
            extra = line_count - _MAX_HISTORY_LINES
            self._transcript_text.delete("1.0", f"{extra + 1}.0")
        self._transcript_text.see(tk.END)  # auto-scroll
        self._transcript_text.configure(state=tk.DISABLED)

        # Clear the partial area now that the utterance is finalized
        self._partial_text.configure(state=tk.NORMAL)
        self._partial_text.delete("1.0", tk.END)
        self._partial_text.configure(state=tk.DISABLED)
