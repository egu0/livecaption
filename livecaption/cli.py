"""Command-line entry point.

Parse args, wire up audio source → ASR → translation → output, wait for Ctrl-C.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console

from . import config
from .asr import AsrWorker, build_recognizer
from .audio import AudioSource, FileSource, MicSource, SystemAudioSource
from .models import resolve_audiotee
from .render import FileWriter, Renderer
from .translate import Translator

# Turn off huggingface_hub download progress bars (they spam the screen even when the model is
# already cached); download failures still raise. hub is lazily imported when each model loads,
# so setting this after the import block is still in time.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


class SourceKind(StrEnum):
    mic = "mic"
    system = "system"
    both = "both"
    file = "file"


class ColorTheme(StrEnum):
    auto = "auto"
    light = "light"
    dark = "dark"


def main(
    source: SourceKind = typer.Option(
        SourceKind.mic,
        help="Audio source: mic / system (meeting output) / both (dual-track) / file",
    ),
    audio_file: Path | None = typer.Option(
        None, "--file", "-f",
        help="Audio file to transcribe (with --source file); wav/mp3/m4a, any sample rate",
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o",
        help="Transcript file path (default: timestamped file under ~/.cache/livecaption)",
    ),
    translate: bool = typer.Option(
        True, "--translate/--no-translate", help="Enable translation"
    ),
    target_lang: str = typer.Option(
        config.DEFAULT_TARGET_LANG, help="Translation target language"
    ),
    asr_model: str = typer.Option(config.DEFAULT_ASR_MODEL, help="ASR model (HF id or local dir)"),
    asr_lang: str = typer.Option(
        config.ASR_LANGUAGE, "--asr-lang",
        help="Spoken language locale, e.g. en-US, ja-JP, de-DE ('auto' = model-detected; "
        "pass an invalid value to list all supported locales)",
    ),
    diarize: bool = typer.Option(
        True, "--diarize/--no-diarize",
        help="Speaker diarization (Sortformer, up to 4 speakers): sentences are "
        "split per speaker and labeled S1/S2/…",
    ),
    diff: bool = typer.Option(
        True, "--diff/--no-diff",
        help="Render final-pass corrections inline: corrected-away words struck "
        "through in red, new words in green",
    ),
    theme: ColorTheme = typer.Option(
        ColorTheme.auto, "--theme",
        help="Terminal color theme: auto (detect background via COLORFGBG, else "
        "fall back to a high-contrast default) / light / dark",
    ),
    mem: bool = typer.Option(
        False, "--mem",
        help="Show MLX unified-memory usage (active/cache/peak) in the bottom status line",
    ),
    mt_model: str = typer.Option(
        config.DEFAULT_MT_MODEL, help="Translation model (HF id or local dir)"
    ),
    context: int = typer.Option(
        config.MT_CONTEXT_SENTENCES,
        help="Prior sentences passed as translation context (0=off; try 2-3)",
    ),
    audiotee: str | None = typer.Option(None, help="Path to the audiotee binary (system source)"),
    include_pid: int | None = typer.Option(
        None, help="Capture only this process PID's audio (e.g. Zoom's PID)"
    ),
    mic_device: str | None = typer.Option(None, help="Microphone device name or index"),
    list_devices: bool = typer.Option(False, help="List audio input devices and exit"),
) -> None:
    console = Console()

    if list_devices:
        import sounddevice as sd

        console.print(sd.query_devices())
        raise typer.Exit()

    # --- build audio sources ---
    need_system = source in (SourceKind.system, SourceKind.both)
    need_mic = source in (SourceKind.mic, SourceKind.both)

    sources: list[AudioSource] = []
    if source is SourceKind.file:
        if audio_file is None or not audio_file.exists():
            console.print("[red]--source file requires an existing --file path[/red]")
            raise typer.Exit(1)
        sources.append(FileSource(str(audio_file), label="file"))
    if need_mic:
        dev: object = mic_device
        if mic_device is not None and mic_device.isdigit():
            dev = int(mic_device)
        sources.append(MicSource(label="me", device=dev))
    if need_system:
        audiotee_bin = resolve_audiotee(audiotee)
        pids = [include_pid] if include_pid is not None else None
        sources.append(SystemAudioSource(audiotee_bin, label="them", include_pids=pids))

    # --- load and warm up the ASR + VAD (+ diarization) models (first run auto-downloads
    # from HF) ---
    try:
        recognizer = build_recognizer(
            asr_model, asr_lang, diarize,
            log=lambda m: console.print(f"[dim]{m}[/dim]"),
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
        raise typer.Exit(130) from None

    if out is None:
        out = (
            Path.home()
            / ".cache"
            / "livecaption"
            / f"transcript-{datetime.now():%Y%m%d-%H%M%S}.txt"
        )
    # meeting transcripts may contain sensitive content; tighten the directory permissions to
    # owner-only
    out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    writer = FileWriter(out, translate=translate)

    stop_event = threading.Event()

    def _on_sigint(*_) -> None:  # noqa: ANN002
        stop_event.set()
        # restore the default handler: a second Ctrl-C raises KeyboardInterrupt to
        # force-interrupt a stuck cleanup
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, _on_sigint)

    with Renderer(
        console=console, theme=theme.value, show_mem=mem, translate=translate
    ) as renderer:
        # --- event callbacks (translator is built after handle_translation, so no placeholder
        # callback) ---
        def handle_translation(label: str, zh_segments: list, started_at: datetime) -> None:
            renderer.translation(label, zh_segments, started_at)
            if writer:
                writer.translation(label, zh_segments, started_at)

        def handle_preview(label: str, zh_segments: list, started_at: datetime) -> None:
            # live provisional translation -> terminal only (never written to the file)
            renderer.preview(label, zh_segments, started_at)

        def handle_translation_disabled() -> None:
            # the model failed to load: release any finals buffered for EN/ZH pairing so they
            # aren't stuck off-screen / unwritten, and fall back to printing EN immediately
            renderer.flush_pending()
            if writer:
                writer.flush_pending()

        translator: Translator | None = None
        if translate:
            translator = Translator(
                mt_model,
                target_lang,
                on_translation=handle_translation,
                on_ready=lambda: console.print("[dim]Translation model ready.[/dim]"),
                on_failed=handle_translation_disabled,
                on_preview=handle_preview,
                context_size=context,
            )
            console.print(f"[dim]Loading translation model {mt_model} …[/dim]")
            translator.start()

        # words already covered by the last live-preview translation, per source (debounce state)
        preview_words: dict[str, int] = {}

        def handle_partial(
            label: str, text: str, started_at: datetime, speaker: int | None = None
        ) -> None:
            renderer.partial(label, text, started_at, speaker)
            # P2 live preview: retranslate the in-progress utterance once it has grown by
            # MT_PREVIEW_MIN_WORDS and contains a sentence-ender (debounced; latest-wins)
            if translator and config.MT_PREVIEW_MIN_WORDS > 0:
                nwords = len(text.split())
                if nwords - preview_words.get(label, 0) >= config.MT_PREVIEW_MIN_WORDS and any(
                    c in text for c in ".?!"
                ):
                    preview_words[label] = nwords
                    translator.submit_preview(label, [(speaker, text, None)], started_at)

        def handle_final(label: str, segments: list, started_at: datetime) -> None:
            preview_words.pop(label, None)  # reset debounce for the next utterance
            # segments = [(speaker, text, diff), ...]; --no-diff drops the two-pass inline
            # correction spans so the terminal shows plain corrected text
            shown = (
                segments
                if diff
                else [(spk, text, None) for spk, text, _d in segments]
            )
            renderer.final(label, shown, started_at)
            if writer:
                # the file always gets the clean, corrected text (diff is ignored there)
                writer.final(label, segments, started_at)
            if translator:
                translator.submit(label, segments, started_at)

        workers: list[AsrWorker] = []
        for src in sources:
            # the weights carry no decode state and can be shared; each worker create_stream's
            # its own independent state.
            # worker crash → stop_event: let the whole pipeline shut down gracefully instead of
            # zombie-living
            worker = AsrWorker(
                recognizer, src.queue, src.label, handle_partial, handle_final,
                on_error=stop_event.set,
            )
            workers.append(worker)
            worker.start()
            src.start()

        labels = " + ".join(s.label for s in sources)
        console.print(
            f"[bold green]● Listening[/bold green]: {labels}  (Ctrl-C to stop)"
        )
        console.print(f"[dim]Transcript → {out}[/dim]")
        if need_system:
            console.print(
                "[dim]Tip: if system audio stays empty, the terminal app likely lacks "
                "Screen & System Audio Recording permission — see README.[/dim]"
            )

        try:
            if source is SourceKind.file:
                # after the file is fully fed (SENTINEL), the worker exits on its own; then
                # drain the translation queue and finish
                for w in workers:
                    while w.is_alive() and not stop_event.is_set():
                        w.join(timeout=0.2)
                if translator is not None and not stop_event.is_set():
                    # None goes at the tail, so the backlog of translations gets done first
                    translator.stop()
                    translator.join()
            else:
                stop_event.wait()
        finally:
            try:
                for src in sources:
                    src.stop()
                for worker in workers:
                    # the finalization path includes a full-sentence two-pass re-decode + a
                    # full-sentence diar feed, so give it plenty of time; if you can't wait,
                    # press Ctrl-C again to force quit
                    worker.join(timeout=15)
                    if worker.is_alive():
                        print(
                            f"\n[warn] ASR worker '{worker.label}' did not finish in "
                            "time; the last sentence may be lost.",
                            file=sys.stderr,
                        )
                if translator is not None:
                    translator.stop()
                    translator.join(timeout=10)
                    if translator.is_alive():
                        print(
                            "\n[warn] pending translations dropped on exit.",
                            file=sys.stderr,
                        )
            except KeyboardInterrupt:
                console.print("\n[red]Force quit; pending output dropped.[/red]")
            finally:
                if writer:
                    writer.close()

    console.print("\n[dim]Stopped.[/dim]")


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
