"""Translation thread: mlx-lm + Hy-MT2, translating finalized sentences serially.

By design only ASR's finalized sentences are translated, never the partials -- a
finalized sentence arriving a second or two late is fine, whereas partials change
constantly, so translating them wastes compute and makes the terminal flicker.
Translation backpressure must never stall audio. Generation goes through
stream_generate and acquires runtime.MLX_LOCK on each decode step: this serializes
all mx evaluation against ASR (MLX gives no guarantees for concurrent evaluation
across threads) while keeping a single translation from freezing the partial for
seconds.
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from collections import deque
from collections.abc import Callable
from datetime import datetime

from . import config
from .runtime import MLX_LOCK


class Translator(threading.Thread):
    """on_translation(label, zh_segments, started_at): zh_segments mirrors the utterance's
    speaker segments as [(speaker, translated_text), ...]; started_at is the utterance start
    time (passed through from submit) so the output side can pair the ZH line with its EN."""

    def __init__(
        self,
        model_name: str,
        target_lang: str,
        on_translation: Callable[[str, list, datetime], None],
        on_ready: Callable[[], None] | None = None,
        on_failed: Callable[[], None] | None = None,
        on_preview: Callable[[str, list, datetime], None] | None = None,
        context_size: int = 0,
    ):
        super().__init__(daemon=True, name="translator")
        self.model_name = model_name
        self.target_lang = target_lang
        self.on_translation = on_translation
        self.on_ready = on_ready
        # called if the model fails to load: lets the output side release any finals it was
        # buffering for pairing (otherwise they'd never get a translation and stay off-screen)
        self.on_failed = on_failed
        # live preview of the in-progress utterance (P2): provisional, latest-wins per source,
        # does not touch the context history; the final re-translation is authoritative
        self.on_preview = on_preview
        self._preview_seq: dict[str, int] = {}
        self._preview_lock = threading.Lock()
        # When context_size > 0, use the most recent N finalized source sentences as
        # translation background (improves reference/terminology coherence)
        self._history: deque[str] | None = (
            deque(maxlen=context_size) if context_size > 0 else None
        )
        self._queue: queue.Queue = queue.Queue()
        self._model = None
        self._tokenizer = None
        self._sampler = None
        self._processors = None

    def submit(self, label: str, segments: list, started_at: datetime) -> None:
        # segments = [(speaker, text, diff), ...]; each speaker segment is translated separately
        # and rejoined into one ZH line so the translation keeps the same inline speaker markers
        with self._preview_lock:  # invalidate any queued preview for this source's utterance
            self._preview_seq[label] = self._preview_seq.get(label, 0) + 1
        self._queue.put(("final", label, segments, started_at))

    def submit_preview(self, label: str, segments: list, started_at: datetime) -> None:
        """Request a provisional translation of the in-progress utterance. Only the newest
        request per source survives -- older queued previews are dropped when dequeued."""
        with self._preview_lock:
            seq = self._preview_seq.get(label, 0) + 1
            self._preview_seq[label] = seq
        self._queue.put(("preview", label, segments, started_at, seq))

    def stop(self) -> None:
        self._queue.put(None)

    def _load(self) -> None:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        # Suppress the harmless "Unrecognized keys in rope_parameters" noise emitted
        # when loading Hy-MT2
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
        except Exception:  # noqa: BLE001
            pass

        self._model, self._tokenizer = load(self.model_name)
        self._sampler = make_sampler(
            temp=config.MT_TEMPERATURE,
            top_p=config.MT_TOP_P,
            top_k=config.MT_TOP_K,
        )
        self._processors = make_logits_processors(
            repetition_penalty=config.MT_REPETITION_PENALTY,
        )

    def _translate(self, text: str, record: bool = True) -> str:
        from mlx_lm import stream_generate

        if self._history:  # only use the background template when there's accumulated context
            # Join with spaces rather than newlines: the "sentences" cut by endpointing are
            # often fragments of a continuous speech stream, and space-joining stays closer
            # to the original continuous text, which works better as background
            content = config.TRANSLATE_PROMPT_WITH_CONTEXT.format(
                context=" ".join(self._history),
                target_lang=self.target_lang,
                text=text,
            )
        else:
            content = config.TRANSLATE_PROMPT.format(
                target_lang=self.target_lang, text=text
            )
        if record and self._history is not None:
            # add the current sentence to history as context for later sentences (previews are
            # provisional, so they don't pollute the context)
            self._history.append(text)
        messages = [{"role": "user", "content": content}]
        # mlx-lm defaults to tokenize=True and returns token ids; matches the official README usage
        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        # Manually drive the generator and acquire MLX_LOCK once per decode step
        # (millisecond granularity): see runtime.py -- holding the lock for a
        # whole-sentence generate would freeze the partial for seconds, so we
        # interleave step by step
        gen = stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=config.MT_MAX_TOKENS,
            sampler=self._sampler,
            logits_processors=self._processors,
        )
        parts: list[str] = []
        while True:
            with MLX_LOCK:
                try:
                    resp = next(gen)
                except StopIteration:
                    break
            parts.append(resp.text)
        return "".join(parts).strip()

    def run(self) -> None:
        try:
            self._load()
        except Exception as e:  # noqa: BLE001
            # Misspelled model name / network down / disk full: clearly state translation
            # is disabled rather than silently vanishing
            print(
                f"\n[error] translation disabled: failed to load {self.model_name}: {e}",
                file=sys.stderr,
            )
            if self.on_failed is not None:
                self.on_failed()
            return
        if self.on_ready:
            self.on_ready()
        backlog_warned = False
        while True:
            item = self._queue.get()
            if item is None:
                break
            if item[0] == "preview":
                _, label, segments, started_at, seq = item
                with self._preview_lock:  # skip if a newer preview / the final superseded it
                    if seq != self._preview_seq.get(label):
                        continue
                if self.on_preview is None:
                    continue
                zh_segments = self._translate_segments(segments, record=False)
                try:
                    self.on_preview(label, zh_segments, started_at)
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
                continue
            _, label, segments, started_at = item
            backlog = self._queue.qsize()
            if backlog >= 10 and not backlog_warned:
                backlog_warned = True
                print(
                    f"\n[warn] translation backlog: {backlog} sentences queued "
                    "(translation is slower than speech; consider a smaller --mt-model)",
                    file=sys.stderr,
                )
            elif backlog == 0:
                backlog_warned = False
            zh_segments = self._translate_segments(segments, record=True)
            try:
                self.on_translation(label, zh_segments, started_at)
            except Exception:  # noqa: BLE001
                # an output-side exception must not kill the translation thread
                traceback.print_exc()

    def _translate_segments(
        self, segments: list, record: bool
    ) -> list[tuple[int | None, str]]:
        out: list[tuple[int | None, str]] = []
        for seg in segments:
            speaker, text = seg[0], seg[1]
            try:
                zh = self._translate(text, record=record)
            except Exception as e:  # noqa: BLE001
                zh = f"[translation failed: {e}]"
            out.append((speaker, zh))
        return out
