"""Consistency check between incremental mel and whole-sentence one-shot computation.

Compares OnlineStream._mel_grow against the whole-sentence one-shot computation.

_mel_grow caches the already-stabilized mel prefix and reruns the STFT only on the new
trailing audio (see comments in asr.py). Here we feed random audio incrementally in random
chunk sizes, comparing step by step against the whole-sentence log_mel_spectrogram result.

Tolerance rationale: on the GPU, inputs of different lengths take different kernel tiling, so
the floating-point summation order for the same frame differs; empirically, "full vs. full
suffix" of the same audio segment already shows ~1.5e-4 absolute difference (a same-shape
recompute is strictly 0) — i.e. the old implementation (full recompute every step, T
constantly changing) carried a difference of this magnitude too. The encoder is bf16
(quantization step ~0.06 at values around 10), so anything below 1e-3 is entirely
imperceptible; a genuine concatenation bug (frame misalignment / contaminated frame entering
the cache) produces an O(1) difference, which this tolerance still reliably catches.
    uv run python scripts/smoke_mel.py
"""

import mlx.core as mx
import numpy as np

from livecaption.asr import build_recognizer
from livecaption.config import DEFAULT_ASR_MODEL

TOL = 1e-3

rec = build_recognizer(DEFAULT_ASR_MODEL)
pre = rec.model.preprocessor_config
print(f"n_fft={pre.n_fft} hop={pre.hop_length} pad_to={getattr(pre, 'pad_to', None)}")

from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram  # noqa: E402

rng = np.random.default_rng(0)
stream = rec.create_stream()
total = np.zeros(0, dtype=np.float32)
worst = 0.0
for i in range(40):
    chunk = (rng.standard_normal(int(rng.integers(800, 4000))) * 0.1).astype(np.float32)
    stream._audio.append(chunk)
    total = np.concatenate([total, chunk])
    inc = stream._mel_grow(final=False)
    ref = log_mel_spectrogram(mx.array(total), pre)
    assert inc.shape == ref.shape, f"step {i}: shape {inc.shape} != {ref.shape}"
    diff = float(mx.abs(inc - ref).max())
    worst = max(worst, diff)
    assert diff < TOL, f"step {i}: max diff {diff}"

inc = stream._mel_grow(final=True)
ref = log_mel_spectrogram(mx.array(total), pre)
assert inc.shape == ref.shape
diff = float(mx.abs(inc - ref).max())
worst = max(worst, diff)
assert diff < TOL, f"final: max diff {diff}"

print(f"worst deviation across {len(total)/16000:.1f}s of audio: {worst:.2e}")
print("incremental mel smoke test PASSED")
