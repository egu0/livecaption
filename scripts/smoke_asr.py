"""Manual smoke test: synthesize English speech with macOS `say`, verify streaming decode.

Verifies the streaming decode against offline generate, three things:
1. build_recognizer can load nemotron-3.5 + Silero VAD (auto-downloaded on first run)
2. OnlineStream fed in 200ms real-time chunks emits partial / final (VAD endpoint
   segmentation works)
3. Streaming final text is largely consistent with offline model.generate (push-style
   stepper bookkeeping is correct)
    uv run python scripts/smoke_asr.py
"""

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from livecaption.asr import build_recognizer
from livecaption.config import ASR_LANGUAGE, DEFAULT_ASR_MODEL, SAMPLE_RATE

TEXT = "The quick brown fox jumps over the lazy dog near the river bank."


def synthesize(text: str, out_wav: Path) -> None:
    """say -> aiff -> afconvert to 16k mono s16 wav (all macOS built-in tools)."""
    aiff = out_wav.with_suffix(".aiff")
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1",
         str(aiff), str(out_wav)],
        check=True,
    )


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE, wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def word_overlap(a: str, b: str) -> float:
    wa = set(a.lower().replace(".", "").replace(",", "").split())
    wb = set(b.lower().replace(".", "").replace(",", "").split())
    return len(wa & wb) / max(len(wa), 1)


with tempfile.TemporaryDirectory() as td:
    wav = Path(td) / "test.wav"
    synthesize(TEXT, wav)
    samples = read_wav(wav)
    print(f"synthesized {len(samples)/SAMPLE_RATE:.1f}s of audio")

    rec = build_recognizer(DEFAULT_ASR_MODEL)
    print("recognizer built OK (nemotron-3.5 + silero-vad)")

    offline = rec.model.generate(str(wav), language=ASR_LANGUAGE).text.strip()
    print(f"offline : {offline}")

    # Simulate real-time: 200ms per chunk, append 3s of silence at the end so rule2
    # endpoint flushes out the last sentence
    stream = rec.create_stream()
    feed = np.concatenate([samples, np.zeros(SAMPLE_RATE * 3, dtype=np.float32)])
    step = int(0.2 * SAMPLE_RATE)
    partials = 0
    finals: list[str] = []
    def _seg_text(segments: list) -> str:  # final events carry [(speaker, text, diff), ...]
        return " ".join(t for _spk, t, _d in segments)

    for i in range(0, len(feed), step):
        for ev in stream.accept_waveform(feed[i : i + step]):
            if ev[0] == "partial":
                partials += 1
            elif ev[0] == "final":
                finals.append(_seg_text(ev[1]))
    finals += [_seg_text(ev[1]) for ev in stream.flush() if ev[0] == "final"]
    print(f"streamed: {' | '.join(finals) if finals else '(no final)'}  "
          f"({partials} partials)")

    assert finals, "streaming produced no final text"
    streamed = " ".join(finals)
    ratio = word_overlap(offline, streamed)
    print(f"word overlap vs offline: {ratio:.0%}")
    assert ratio >= 0.6, f"streamed text diverges from offline ({ratio:.0%})"
    assert word_overlap(TEXT, streamed) >= 0.5, "streamed text diverges from spoken text"

print("ASR smoke test PASSED")
