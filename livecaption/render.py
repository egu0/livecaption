"""Output layer: terminal rendering (rich) and file writing, both thread-safe.

A finalized sentence and its translation are committed together as a pair, so each EN
line is immediately followed by its own ZH line. Because translation lags behind
speech, the finalized EN is buffered until its ZH arrives, then the EN+ZH pair is
committed at once (printed to the terminal / written to the file). The translator is a
single FIFO thread, so translations arrive in finalization order and the buffer flushes
completed entries from the head, keeping sentence order intact even when an earlier
translation is still in flight. The bottom active area keeps showing the live partial
meanwhile. With --no-translate (or if the translation model fails to load) the EN line
is emitted immediately. Any still-buffered finals are flushed on shutdown.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.text import Text

# Fixed styles that stay neutrally visible on both backgrounds: avoid dim (dim lowers
# brightness relative to the background, washing toward white on light backgrounds and toward
# black on dark ones, faint either way). grey50 is an absolute mid-gray, visible on black and
# white alike; red/magenta are among the few colors safe on both backgrounds. Only the
# translation truly varies with the background (see _TRANSLATION_STYLE).
_COMMON_STYLES = {
    # timestamp: secondary info, mid-gray de-emphasized but not invisible
    "ts": "grey50",
    # partial result: weaker than final (it'll be overwritten); mid-gray signals "tentative"
    "partial": "grey50",
    # finalized source text: terminal's default foreground (the terminal theme guarantees
    # readability)
    "final": "",
    "diff_del": "bold red strike",
    "diff_add": "bold green",
    # MLX memory status line: secondary diagnostic info, same neutral mid-gray as partials
    "mem": "grey50",
}
# The translation is what the user cares about most, so it's tuned per background:
# - default: don't gamble on a color; use the default foreground + bold, relying on weight +
#   Chinese characters to distinguish it from the English source, clear on any background
# - dark/light: keep a color distinction, using bright cyan / deep cyan-blue respectively
_TRANSLATION_STYLE = {
    "default": "bold",
    "dark": "bold bright_cyan",
    "light": "deep_sky_blue4",
}
# Speaker palette (Sortformer supports up to 4 people): one color each for S1-S4, so a
# multi-person meeting is readable at a glance; the 5th person onward falls back to the last
# color. Also split per background -- any single fixed palette leaves someone invisible on some
# background.
# Color is just a nice-to-have; the "S1"/"S2" text alone already distinguishes them, so default
# just uses base colors.
# Note: deliberately avoid the cyan/cyan-blue family, otherwise it would clash with the
# translation (dark=bright_cyan / light=deep_sky_blue4)
_SPEAKER_PALETTE = {
    "default": ["bold magenta", "bold blue", "bold green", "bold red"],
    "dark": ["bold bright_magenta", "bold bright_yellow", "bold bright_green", "bold bright_red"],
    "light": ["bold magenta", "bold dark_orange3", "bold green4", "bold red"],
}


# Safety valve: how many finalized-but-uncommitted utterances may sit in the live region before
# we commit the oldest early (with whatever ZH is available). Stops a translation backlog from
# growing the live region until it fills the terminal and breaks rich's scroll-above-Live.
_MAX_PENDING = 8


def _segments_text(segments: list) -> str:
    """Flatten speaker segments to plain text with inline [S1]/[S2] markers, for the file.
    Each segment is (speaker, text, ...); speaker None (diarize off) drops the marker."""
    parts = []
    for seg in segments:
        speaker, text = seg[0], seg[1]
        parts.append(f"[S{speaker + 1}] {text}" if speaker is not None else text)
    return "  ".join(parts)


def _resolve_theme(theme: str) -> str:
    """Resolve the color scheme: light/dark/default are taken directly; auto tries to detect
    background lightness from COLORFGBG, and when it can't (most macOS terminals don't set this
    variable) it falls back to default -- the translation uses default foreground + bold, which
    is always readable."""
    if theme in ("light", "dark", "default"):
        return theme
    bg = os.environ.get("COLORFGBG", "").split(";")[-1].strip()
    # the last segment of COLORFGBG is the background color index; 7/15 = light background
    if bg in ("7", "15"):
        return "light"
    if bg.isdigit():
        return "dark"
    return "default"


class Renderer:
    def __init__(
        self,
        console: Console | None = None,
        theme: str = "auto",
        show_mem: bool = False,
        translate: bool = True,
    ):
        self.console = console or Console()
        resolved = _resolve_theme(theme)
        self._sty = {**_COMMON_STYLES, "translation": _TRANSLATION_STYLE[resolved]}
        self._palette = _SPEAKER_PALETTE[resolved]
        # source label ("me"/"them") -> (started_at, partial text, speaker, zh_preview) for the
        # in-progress utterance, or None when idle. zh_preview is the live (debounced) provisional
        # translation shown under the EN partial (P2); speaker is the diar peek's current guess.
        self._partials: dict[str, tuple[datetime, str, int | None, Text | None] | None] = {}
        self._lock = threading.Lock()
        # Finalized utterances awaiting their final (two-pass) translation. They stay VISIBLE in
        # the live region (EN final + the carried-over zh_preview) so the line refreshes in place
        # rather than vanishing then reappearing; once the final ZH arrives the EN+ZH pair is
        # committed to history (scrolls up) in finalization order. Entries:
        # {label, started_at, en: Text, zh: Text | None, zh_preview: Text | None}.
        self._translate = translate
        self._pending: list[dict] = []
        # optional MLX unified-memory readout in the bottom status line, refreshed by a
        # background thread so it ticks even during silence (off by default)
        self._show_mem = show_mem
        self._mx = None
        if show_mem:
            import mlx.core as mx

            self._mx = mx
        self._mem_stop = threading.Event()
        self._mem_thread: threading.Thread | None = None
        from rich.live import Live

        self._live = Live(
            Text(""),
            console=self.console,
            refresh_per_second=12,
            transient=False,
        )

    def __enter__(self) -> Renderer:
        self._live.start()
        if self._show_mem:
            self._mem_thread = threading.Thread(
                target=self._mem_loop, daemon=True, name="mem-monitor"
            )
            self._mem_thread.start()
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self._mem_stop.set()
        if self._mem_thread is not None:
            self._mem_thread.join(timeout=2)
        with self._lock:
            self._flush_pending_locked()
            self._partials.clear()
            self._live.update(Text(""))
        self._live.stop()

    def _speaker_style(self, spk: str) -> str:
        """Pick a color by speaker number ("S1" -> 1st color ...; the 5th person onward falls
        back to the last color)."""
        try:
            idx = int(spk[1:]) - 1
        except (ValueError, IndexError):
            return self._palette[0]
        return self._palette[min(max(idx, 0), len(self._palette) - 1)]

    def _render_active(self):
        lines = []
        # finalized utterances still waiting on their final translation: shown with the
        # provisional preview ZH (if any) so the line is stable, until the final ZH commits it
        for e in self._pending:
            lines.append(e["en"])
            zh = e["zh"] or e["zh_preview"]
            if zh is not None:
                lines.append(zh)
        for v in self._partials.values():
            if not v:
                continue
            started_at, text, speaker, zh_preview = v
            if not text:
                continue
            line = Text(f"[{started_at:%H:%M:%S}] ", style=self._sty["ts"])
            if speaker is not None:
                line.append(f"[S{speaker + 1}] ", style=self._speaker_style(f"S{speaker + 1}"))
            line.append(text, style=self._sty["partial"])
            lines.append(line)
            if zh_preview is not None:  # live provisional translation under the EN partial
                lines.append(zh_preview)
        if self._show_mem:
            lines.append(self._mem_line())
        return Group(*lines) if lines else Text("")

    def _append_segments(self, line: Text, segments: list, text_style: str) -> None:
        """Append speaker segments to `line` with inline [S1]/[S2] markers (in the speaker's
        colour), so a multi-speaker utterance stays a single line instead of one line each.

        segments are (speaker, text, diff); for the EN line diff is the two-pass inline-diff
        spans (struck-out / added words) and overrides the per-word style, otherwise the whole
        segment is text_style. speaker None (diarize off) means no marker.
        """
        for i, (speaker, text, diff) in enumerate(segments):
            if i:
                line.append("  ")
            if speaker is not None:
                line.append(f"[S{speaker + 1}] ", style=self._speaker_style(f"S{speaker + 1}"))
            if diff:
                styles = {
                    "same": text_style,
                    "del": self._sty["diff_del"],
                    "add": self._sty["diff_add"],
                }
                for j, (kind, words) in enumerate(diff):
                    if j:
                        line.append(" ")
                    line.append(words, style=styles.get(kind, ""))
            else:
                line.append(text, style=text_style)

    def _mem_line(self) -> Text:
        """One status line of MLX unified-memory usage. active = buffers MLX currently holds,
        cache = freed buffers MLX keeps for reuse, peak = high-water mark since start. The
        getters are lightweight counter reads (no mx evaluation), so they need no MLX_LOCK."""
        g = 1 / 1e9
        return Text(
            f"MLX  active {self._mx.get_active_memory() * g:.2f} GB"
            f" · cache {self._mx.get_cache_memory() * g:.2f} GB"
            f" · peak {self._mx.get_peak_memory() * g:.2f} GB",
            style=self._sty["mem"],
        )

    def _mem_loop(self) -> None:
        """Refresh the status line on a timer so the memory readout ticks even during silence
        (partial-driven updates only fire while someone is speaking)."""
        while not self._mem_stop.wait(1.0):
            with self._lock:
                self._live.update(self._render_active())

    def partial(
        self, label: str, text: str, started_at: datetime, speaker: int | None = None
    ) -> None:
        with self._lock:
            prev = self._partials.get(label)
            # keep the live preview ZH while the same utterance keeps growing
            zh_preview = prev[3] if (prev and prev[0] == started_at) else None
            self._partials[label] = (started_at, text, speaker, zh_preview)
            self._live.update(self._render_active())

    def preview(self, label: str, zh_segments: list, started_at: datetime) -> None:
        """Live provisional translation of the in-progress utterance (P2): shown under the EN
        partial, replaced by the final translation later. zh_segments = [(speaker, zh), ...]."""
        line = Text()
        line.append(f"[{started_at:%H:%M:%S}] ", style=self._sty["ts"])
        self._append_segments(
            line, [(spk, zh, None) for spk, zh in zh_segments], self._sty["translation"]
        )
        with self._lock:
            prev = self._partials.get(label)
            if prev and prev[0] == started_at:
                self._partials[label] = (prev[0], prev[1], prev[2], line)
            else:
                # the utterance finalized before this preview returned -> attach to its pending
                # entry so the finalized line still shows a provisional ZH until the final one
                for e in self._pending:
                    if (
                        e["zh"] is None
                        and e["label"] == label
                        and e["started_at"] == started_at
                    ):
                        e["zh_preview"] = line
                        break
            self._live.update(self._render_active())

    def final(self, label: str, segments: list, started_at: datetime) -> None:
        # segments = [(speaker, text, diff), ...] for the whole utterance; rendered as one line
        # with inline [S1]/[S2] markers (see _append_segments). Note: Text(str, style=...) sets
        # a whole-line base style that would mask later appends, so build empty + append.
        line = Text()
        line.append(f"[{started_at:%H:%M:%S}] ", style=self._sty["ts"])
        self._append_segments(line, segments, self._sty["final"])
        with self._lock:
            prev = self._partials.get(label)
            zh_preview = prev[3] if (prev and prev[0] == started_at) else None
            self._partials[label] = None
            if self._translate:
                # keep the finalized line visible (with its preview ZH) until the final ZH
                # arrives, then commit the pair to history
                self._pending.append(
                    {
                        "label": label,
                        "started_at": started_at,
                        "en": line,
                        "zh": None,
                        "zh_preview": zh_preview,
                    }
                )
                self._relieve_pending_locked()
            else:
                self._live.console.print(line)
            self._live.update(self._render_active())

    def _print_translation(self, line: Text) -> None:
        # soft_wrap so the terminal hard-wraps the line itself. Chinese has no spaces, so rich's
        # word wrapper treats the whole translation as one giant unbreakable "word": when it
        # doesn't fit after the speaker label it bumps the entire run to the next line before
        # folding (looks like an empty "[ts] S1" line followed by the text). Letting the terminal
        # wrap it character by character -- which is how CJK wraps anyway -- keeps it on the same
        # line as the label. (English keeps rich's nicer word wrapping; it has spaces.)
        self._live.console.print(line, soft_wrap=True)

    def translation(self, label: str, zh_segments: list, started_at: datetime) -> None:
        # zh_segments = [(speaker, zh_text), ...] mirrors the EN segments, so the ZH line carries
        # the same inline [S1]/[S2] markers
        line = Text()
        line.append(f"[{started_at:%H:%M:%S}] ", style=self._sty["ts"])
        self._append_segments(
            line, [(spk, zh, None) for spk, zh in zh_segments], self._sty["translation"]
        )
        with self._lock:
            if not self._translate:
                self._print_translation(line)
                return
            # attach this ZH to its buffered EN (matched by label + start time), then commit
            # completed pairs from the head so sentences stay in order even if an earlier
            # translation is still in flight
            for entry in self._pending:
                if (
                    entry["zh"] is None
                    and entry["label"] == label
                    and entry["started_at"] == started_at
                ):
                    entry["zh"] = line
                    break
            else:
                # no matching entry: it was already committed early by the height valve (with
                # its preview ZH), so drop this late final re-translation rather than printing a
                # stray out-of-order line
                return
            while self._pending and self._pending[0]["zh"] is not None:
                done = self._pending.pop(0)
                self._live.console.print(done["en"])
                self._print_translation(done["zh"])
            self._live.update(self._render_active())

    def _relieve_pending_locked(self) -> None:
        """Height valve: if the live region is holding too many finalized-but-untranslated
        utterances (translation backlog), commit the oldest early using whatever ZH is on hand
        (final, else the provisional preview) so the region can't grow until it fills the
        terminal. Caller holds the lock."""
        while len(self._pending) > _MAX_PENDING:
            done = self._pending.pop(0)
            self._live.console.print(done["en"])
            zh = done["zh"] or done["zh_preview"]
            if zh is not None:
                self._print_translation(zh)

    def _flush_pending_locked(self) -> None:
        """Commit every buffered final (with its ZH, else its provisional preview) in order.
        Caller holds the lock. Also stops buffering, so later finals print immediately -- used
        on shutdown and when translation is disabled mid-run (model failed to load)."""
        for entry in self._pending:
            self._live.console.print(entry["en"])
            zh = entry["zh"] or entry["zh_preview"]
            if zh is not None:
                self._print_translation(zh)
        self._pending.clear()
        self._translate = False

    def flush_pending(self) -> None:
        """Release buffered finals so they aren't stuck off-screen when translation is disabled
        at runtime (e.g. the model failed to load)."""
        with self._lock:
            self._flush_pending_locked()
            self._live.update(self._render_active())


class FileWriter:
    """Append-writes the transcript file. Each finalized EN line is buffered and written
    together with its ZH translation (indented), so the file reads as EN/ZH pairs in sentence
    order rather than a batch of EN lines followed by a batch of ZH lines. The translator is
    FIFO, so translations arrive in order and completed entries flush from the head of the
    buffer. With translate=False the EN line is written immediately; any still-buffered finals
    are flushed on close. Late events arriving after close are simply dropped (a worker may
    briefly survive past the cleanup-path join timeout; see cli)."""

    def __init__(self, path: Path, translate: bool = True):
        self._f = open(path, "a", encoding="utf-8")  # noqa: SIM115
        self._lock = threading.Lock()
        self._translate = translate
        # buffered finals awaiting translation, kept in finalization order; entries:
        # {label, started_at, en: str, zh: str | None}
        self._pending: list[dict] = []

    def final(self, label: str, segments: list, started_at: datetime) -> None:
        en = f"[{started_at:%H:%M:%S}] [{label}] {_segments_text(segments)}\n"
        with self._lock:
            if self._f.closed:
                return
            if not self._translate:
                self._f.write(en)
                self._f.flush()
                return
            self._pending.append(
                {"label": label, "started_at": started_at, "en": en, "zh": None}
            )

    def translation(self, label: str, zh_segments: list, started_at: datetime) -> None:
        zh_line = f"    [{started_at:%H:%M:%S}] [{label}] {_segments_text(zh_segments)}\n"
        with self._lock:
            if self._f.closed:
                return
            if not self._translate:
                self._f.write(zh_line)
                self._f.flush()
                return
            for entry in self._pending:
                if (
                    entry["zh"] is None
                    and entry["label"] == label
                    and entry["started_at"] == started_at
                ):
                    entry["zh"] = zh_line
                    break
            else:
                self._f.write(zh_line)  # no buffered source (already flushed): write alone
                self._f.flush()
                return
            while self._pending and self._pending[0]["zh"] is not None:
                done = self._pending.pop(0)
                self._f.write(done["en"])
                self._f.write(done["zh"])
            self._f.flush()

    def _flush_pending_locked(self) -> None:
        """Write out every buffered final (with its ZH if present) in order, then stop
        buffering. Caller holds the lock."""
        for entry in self._pending:
            self._f.write(entry["en"])
            if entry["zh"] is not None:
                self._f.write(entry["zh"])
        self._pending.clear()
        self._translate = False

    def flush_pending(self) -> None:
        """Release buffered finals when translation is disabled at runtime (model load failed)."""
        with self._lock:
            if self._f.closed:
                return
            self._flush_pending_locked()
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._flush_pending_locked()
                self._f.flush()
            self._f.close()
