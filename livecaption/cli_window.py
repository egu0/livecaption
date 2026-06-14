"""CLI entry point for the system-audio window mode.

Reuses the existing audio/ASR pipeline but forces fixed defaults:
  - Source: system audio only
  - Translation: disabled
  - Diarization: disabled
  - File output: disabled

The window renderer replaces the terminal renderer; everything else is identical
to the main CLI pipeline (SystemAudioSource + AsrWorker).
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading

import typer

from . import config
from .asr import AsrWorker, build_recognizer
from .audio import SystemAudioSource
from .models import resolve_audiotee, resolve_caption_window
from .swift_window import SwiftCaptionWindow

# Turn off huggingface_hub download progress bars (they spam the terminal even when
# the model is already cached); download failures still raise.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def main(
    asr_model: str = typer.Option(
        config.DEFAULT_ASR_MODEL, help="ASR model (HF id or local dir)"
    ),
    asr_lang: str = typer.Option(
        config.ASR_LANGUAGE, "--asr-lang",
        help="Spoken language locale, e.g. en-US, ja-JP, de-DE",
    ),
    audiotee: str | None = typer.Option(
        None, help="Path to the audiotee binary"
    ),
    include_pid: int | None = typer.Option(
        None, help="Capture only this process PID's audio (e.g. Zoom's PID)"
    ),
    caption_window: str | None = typer.Option(
        None, "--caption-window",
        help="Path to the livecaption-window binary (default: auto-resolve from ./bin or PATH)",
    ),
) -> None:
    """LiveCaption Window — system audio transcription in a desktop window."""
    stop_event = threading.Event()
    window: SwiftCaptionWindow | None = None
    worker: AsrWorker | None = None
    source: SystemAudioSource | None = None

    def _on_signal(*_: object) -> None:
        """Ctrl-C in the terminal requests graceful shutdown."""
        stop_event.set()
        # Schedule Tk shutdown from the main thread; if the window is already gone,
        # root.quit via after is harmless
        if window is not None:
            with contextlib.suppress(Exception):
                window.close()
        # Restore default handler: a second Ctrl-C force-exits
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, _on_signal)

    # ---- Phase 1: Resolve Swift window binary ----
    try:
        caption_bin = resolve_caption_window(caption_window)
    except FileNotFoundError as e:
        print(f"Fatal: {e}", file=sys.stderr)
        raise typer.Exit(1) from None

    # ---- Phase 2: Create window (native Swift via subprocess) ----
    window = SwiftCaptionWindow(caption_bin)
    window.set_stop_event(stop_event)
    window.set_status("Loading ASR model…")

    # ---- Phase 3: Resolve audiotee ----
    try:
        audiotee_bin = resolve_audiotee(audiotee)
    except FileNotFoundError as e:
        window.set_status(f"Fatal: {e}")
        print(f"Fatal: {e}", file=sys.stderr)
        window.show()  # let the user see the error before closing
        raise typer.Exit(1) from None
    except Exception as e:  # noqa: BLE001
        window.set_status(f"Fatal: {e}")
        print(f"Fatal: {e}", file=sys.stderr)
        window.show()
        raise typer.Exit(1) from None

    # ---- Phase 4: Load ASR model ----
    try:
        recognizer = build_recognizer(
            asr_model, asr_lang, diarize=False,
            log=lambda m: window.set_status(f"ASR: {m}"),
        )
    except ValueError as e:
        window.set_status(f"Fatal: {e}")
        print(f"Fatal: {e}", file=sys.stderr)
        window.show()
        raise typer.Exit(1) from None

    # ---- Phase 5: Build pipeline ----
    pids = [include_pid] if include_pid is not None else None
    source = SystemAudioSource(audiotee_bin, label="them", include_pids=pids)

    def _on_error() -> None:
        """ASR worker crashed — stop the pipeline so the window doesn't pretend to listen."""
        window.set_status("Error: ASR worker stopped unexpectedly")
        stop_event.set()
        if window is not None:
            with contextlib.suppress(Exception):
                window.close()

    worker = AsrWorker(
        recognizer,
        source.queue,
        source.label,
        on_partial=window.partial,
        on_final=window.final,
        on_error=_on_error,
    )

    # ---- Phase 6: Start pipeline ----
    source.start()
    worker.start()
    window.set_status("● Listening")

    # ---- Phase 7: Run until stop ----
    # The window main loop blocks here. The window-close button sets stop_event
    # and calls root.quit(), which unblocks mainloop.
    window.show()

    # ---- Phase 8: Cleanup ----
    window.set_status("Stopping…")
    if source is not None:
        source.stop()
    if worker is not None:
        worker.join(timeout=15)
        if worker.is_alive():
            print(
                "\n[warn] ASR worker did not finish in time; "
                "the last sentence may be lost.",
                file=sys.stderr,
            )


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
