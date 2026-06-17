#!/usr/bin/env python3
"""Spike: is mlx-qwen3-asr a viable Chinese streaming backend for livecaption?

De-risks the four unknowns BEFORE committing to the `--asr-backend nemotron|qwen3` work:
  1. Chinese accuracy  -- read the transcript; optional nemotron side-by-side
  2. Partial quality   -- streaming_metrics rewrite_rate / partial_stability (tail flicker)
  3. Real-time factor  -- decode time vs audio length, per-chunk decode latency
  4. Diarization loss  -- (manual) the streaming path has NO speakers / NO timestamps;
                          just eyeball whether that hurts for your meetings

It only exercises offline transcribe() + the rolling feed_audio() loop -- exactly the seam
a `--asr-backend qwen3` OnlineStream adapter would wrap. No mic / sounddevice needed.

Run WITHOUT touching project deps (uv installs the spike-only package transiently):

    uv run --with mlx-qwen3-asr python scripts/spike_qwen3.py CHINESE.wav
    uv run --with mlx-qwen3-asr python scripts/spike_qwen3.py                 # downloads official zh sample
    uv run --with mlx-qwen3-asr python scripts/spike_qwen3.py CHINESE.wav --model mlx-community/Qwen3-ASR-1.7B-8bit
    uv run --with mlx-qwen3-asr python scripts/spike_qwen3.py CHINESE.wav --nemotron-lang zh-cn   # side-by-side

Prefer a REAL Chinese meeting recording over the bundled sample -- that is the case we
actually care about (spontaneous speech, code-switching, jargon).
"""

# ruff: noqa: E501 -- dev spike: copy-paste example commands + aligned diagnostic prints
from __future__ import annotations

import argparse
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
ZH_SAMPLE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
FEED_CHUNK_SEC = 0.48  # simulate a mic blocksize; feed_audio() buffers until chunk_size_sec


def load_audio_16k(path: str | None) -> tuple[np.ndarray, str]:
    """Return (float32 mono 16k array, source path). Downloads the official zh sample if path is None."""
    from mlx_qwen3_asr import load_audio

    if path is None:
        dst = Path(tempfile.gettempdir()) / "qwen3_asr_zh_sample.wav"
        if not dst.exists():
            print(f"No audio given -> downloading official Chinese sample to {dst}")
            urllib.request.urlretrieve(ZH_SAMPLE_URL, dst)  # noqa: S310 (trusted Qwen OSS url)
        path = str(dst)
    audio = np.asarray(load_audio(path), dtype=np.float32)
    return audio, path


def char_overlap(a: str, b: str) -> float:
    """Rough agreement for CJK text (no word boundaries): Jaccard over char sets."""
    sa, sb = set(a.strip()), set(b.strip())
    return len(sa & sb) / max(len(sa | sb), 1)


def run_offline(model: str, audio: np.ndarray, language: str | None, dur: float) -> str:
    from mlx_qwen3_asr import transcribe

    # Warm up: the FIRST call pays a one-time cost (model download on first ever run +
    # weight load + Metal kernel compile) that has nothing to do with steady-state speed.
    # Time only the second (warm) call so the reported RTF reflects real inference.
    print("  warming up (one-time model download/load/compile, not timed)...")
    transcribe(audio[:SAMPLE_RATE], model=model, language=language)
    t0 = time.perf_counter()
    result = transcribe(audio, model=model, language=language)
    dt = time.perf_counter() - t0
    print(f"  detected language : {result.language}")
    print(f"  decode time       : {dt:.2f}s    RTF: {dt / dur:.3f}x")
    print(f"  text              : {result.text!r}\n")
    return result.text


def run_streaming(model: str, audio: np.ndarray, language: str | None,
                  chunk_size_sec: float, mode: str, dur: float) -> str:
    from mlx_qwen3_asr.streaming import (
        feed_audio,
        finish_streaming,
        init_streaming,
        streaming_metrics,
    )

    state = init_streaming(
        model=model,
        chunk_size_sec=chunk_size_sec,
        max_context_sec=30.0,
        language=language,
        finalization_mode=mode,
    )

    feed = int(FEED_CHUNK_SEC * SAMPLE_RATE)
    last = ""
    decode_times: list[float] = []  # per feed call that actually ran a chunk decode

    print("  (partial trace -- APPEND = clean growth, REWRITE = tail rewrote prior text)")
    t_start = time.perf_counter()
    for i in range(0, len(audio), feed):
        sub = audio[i : i + feed]
        prev_chunk_id = state.chunk_id
        t0 = time.perf_counter()
        state = feed_audio(sub, state)
        dt = time.perf_counter() - t0
        if state.chunk_id > prev_chunk_id:
            decode_times.append(dt)
        if state.text != last:
            tag = "APPEND " if state.text.startswith(last) else "REWRITE"
            print(f"    [{i / SAMPLE_RATE:6.2f}s] {tag} {state.text!r}")
            last = state.text
    state = finish_streaming(state)
    total = time.perf_counter() - t_start

    m = streaming_metrics(state)
    max_decode = max(decode_times) if decode_times else 0.0
    mean_decode = (sum(decode_times) / len(decode_times)) if decode_times else 0.0

    print(f"\n  final text        : {state.text!r}")
    print(f"  metrics           : {m}")
    print(f"  streaming RTF     : {total / dur:.3f}x  (total {total:.2f}s for {dur:.1f}s audio)")
    print(f"  per-chunk decode  : mean {mean_decode * 1000:.0f}ms  max {max_decode * 1000:.0f}ms"
          f"  (chunk budget = {chunk_size_sec * 1000:.0f}ms)")
    keeps_up = max_decode < chunk_size_sec and (total / dur) < 1.0
    print(f"  keeps up live?    : {'YES' if keeps_up else 'NO -- decode slower than real time'}"
          "   [note: solo run; real pipeline shares the GPU/MLX_LOCK with Hy-MT2 translation]")
    print(f"  partial_stability : {m['partial_stability']:.0%} committed   "
          f"rewrite_rate: {m['rewrite_rate']:.0%}  "
          f"(<- high rewrite_rate = flickery tail)\n")
    return state.text


def run_nemotron(path: str, language: str) -> str | None:
    """Offline nemotron on the SAME audio for an accuracy baseline. Returns None on failure
    (e.g. nemotron doesn't support this language -- the error lists supported keys)."""
    try:
        from livecaption.asr import build_recognizer
        from livecaption.config import DEFAULT_ASR_MODEL
    except Exception as e:  # noqa: BLE001
        print(f"  (skipped: cannot import livecaption -- run from repo root) {e}\n")
        return None
    try:
        t0 = time.perf_counter()
        rec = build_recognizer(DEFAULT_ASR_MODEL, language=language)
        text = rec.model.generate(str(path), language=rec.language).text.strip()
        dt = time.perf_counter() - t0
        print(f"  nemotron ({language}) decode {dt:.2f}s")
        print(f"  text              : {text!r}\n")
        return text
    except Exception as e:  # noqa: BLE001
        print(f"  nemotron comparison FAILED for language={language!r}:\n    {e}\n")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", default=None, help="path to a 16k-ish wav/audio; omit to fetch zh sample")
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B",
                    help="moona3k-compatible model: an HF fp16 id (Qwen/Qwen3-ASR-1.7B | Qwen/Qwen3-ASR-0.6B) "
                         "or a LOCAL dir produced by mlx-qwen3-asr's own converter. NOTE: the "
                         "mlx-community/Qwen3-ASR-*-8bit repos are mlx-audio's and do NOT load in this package.")
    ap.add_argument("--language", default="Chinese", help="Qwen3 language name (e.g. Chinese, English)")
    ap.add_argument("--chunk-size-sec", type=float, default=2.0)
    ap.add_argument("--mode", choices=["accuracy", "latency"], default="accuracy")
    ap.add_argument("--nemotron-lang", default=None,
                    help="if set, run livecaption's nemotron offline with this lang key for side-by-side")
    args = ap.parse_args()

    audio, path = load_audio_16k(args.audio)
    dur = len(audio) / SAMPLE_RATE
    print(f"\nAudio : {path}  ({dur:.1f}s, {len(audio)} samples @ {SAMPLE_RATE}Hz)")
    print(f"Model : {args.model}   language={args.language}   chunk={args.chunk_size_sec}s   mode={args.mode}")

    print("\n" + "=" * 74)
    print("1) OFFLINE transcribe()  -- accuracy ceiling + RTF baseline")
    print("=" * 74)
    offline_text = run_offline(args.model, audio, args.language, dur)

    print("=" * 74)
    print("2) STREAMING feed_audio() loop  -- what a live --asr-backend qwen3 would emit")
    print("=" * 74)
    stream_text = run_streaming(args.model, audio, args.language, args.chunk_size_sec, args.mode, dur)

    print("=" * 74)
    print("3) AGREEMENT  streaming-final vs offline")
    print("=" * 74)
    print(f"  char-set Jaccard  : {char_overlap(offline_text, stream_text):.0%}"
          "  (low = streaming tail drops/garbles vs offline)\n")

    if args.nemotron_lang:
        print("=" * 74)
        print("4) NEMOTRON baseline (same audio, offline) -- is Chinese actually worse?")
        print("=" * 74)
        nemo_text = run_nemotron(path, args.nemotron_lang)
        if nemo_text is not None:
            print(f"  qwen3 vs nemotron char-set Jaccard: {char_overlap(offline_text, nemo_text):.0%}")
            print("  -> read both transcripts above; pick whichever is actually correct Chinese.\n")

    print("Done. Judge: (a) is Chinese accuracy worth it, (b) is the partial tail too flickery,")
    print("(c) does it keep up live, (d) does losing speakers/timestamps hurt your meetings?")


if __name__ == "__main__":
    main()
