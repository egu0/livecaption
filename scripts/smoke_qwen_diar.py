"""Pure-logic smoke test for the qwen3 word-level diarization split (no model, runs in
seconds).

Verifies _word_offsets (locating aligned words in the canonical text) and
_segments_from_speaker_seq (slicing the text at speaker-change word boundaries, verbatim).
These are the risky parts of the forced-aligner-driven mid-sentence speaker split; the
Sortformer frame->speaker mapping itself needs the model and is exercised by a real run.
    uv run python scripts/smoke_qwen_diar.py
"""

from livecaption.asr_qwen import _segments_from_speaker_seq, _word_offsets


def chars(text: str) -> list[dict]:
    # CJK aligner output: one bare char per word, 0.3s each
    return [{"text": c, "start": i * 0.3, "end": i * 0.3 + 0.3} for i, c in enumerate(text)]


# --- _word_offsets ---

# 1. CJK char tokens map to their own index (contiguous, no spaces)
text = "你好世界"
assert _word_offsets(chars(text), text) == [(0, 1), (1, 2), (2, 3), (3, 4)]

# 2. Duplicate tokens resolve in order (search advances, never backtracks)
t = "the cat the dog"
ws = [{"text": w, "start": i, "end": i + 1} for i, w in enumerate(t.split())]
assert _word_offsets(ws, t) == [(0, 3), (4, 7), (8, 11), (12, 15)], _word_offsets(ws, t)

# 3. A word absent from the text (e.g. punctuation the aligner stripped) collapses to a
#    zero-width offset at the cursor and doesn't move it backwards
ws = [{"text": "好", "start": 0, "end": 1}, {"text": "X", "start": 1, "end": 2},
      {"text": "的", "start": 2, "end": 3}]
assert _word_offsets(ws, "好的") == [(0, 1), (1, 1), (1, 2)]


# --- _segments_from_speaker_seq ---

# 4. CJK two speakers: clean split, concatenation reproduces the text
text = "我觉得可以那就这么定"  # 10 chars: 我觉得可以 | 那就这么定
w = chars(text)
spk = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # change at index 5 ("那")
segs = _segments_from_speaker_seq(w, spk, text)
assert segs == [(0, "我觉得可以"), (1, "那就这么定")], segs
assert "".join(s for _k, s in segs) == text

# 5. English two speakers: each segment is a verbatim slice (inter-speaker space dropped by strip)
text = "hello world how are you"
w = [{"text": x, "start": i, "end": i + 1} for i, x in enumerate(text.split())]
spk = [0, 0, 1, 1, 1]
assert _segments_from_speaker_seq(w, spk, text) == [(0, "hello world"), (1, "how are you")]

# 6. Mixed CJK + Latin: spacing around embedded Latin is preserved inside each segment
text = "我们用 GPU 跑 training 任务"
# aligner tokenization: CJK per char, Latin per word
w = [{"text": x, "start": i, "end": i + 1}
     for i, x in enumerate(["我", "们", "用", "GPU", "跑", "training", "任", "务"])]
spk = [0, 0, 0, 0, 0, 1, 1, 1]  # change at "training"
segs = _segments_from_speaker_seq(w, spk, text)
assert segs == [(0, "我们用 GPU 跑"), (1, "training 任务")], segs

# 7. Sentence-final punctuation stays with the preceding speaker (it sits before the next
#    group's first word in the text)
text = "好的。明白了"
w = [{"text": x, "start": i, "end": i + 1} for i, x in enumerate(["好", "的", "明", "白", "了"])]
spk = [0, 0, 1, 1, 1]
assert _segments_from_speaker_seq(w, spk, text) == [(0, "好的。"), (1, "明白了")]

# 8. Single speaker -> one segment equal to the whole text
text = "全程只有一个人在说话"
w = chars(text)
assert _segments_from_speaker_seq(w, [0] * len(text), text) == [(0, text)]

# 9. No words -> single unlabeled segment (defensive)
assert _segments_from_speaker_seq([], [], "whatever") == [(None, "whatever")]

# 10. A leading short run still anchors the first segment at offset 0 (no leading text lost)
text = "嗯我同意你的看法"
w = chars(text)
spk = [0, 1, 1, 1, 1, 1, 1, 1]  # "嗯" is speaker 0, rest speaker 1
segs = _segments_from_speaker_seq(w, spk, text)
assert segs == [(0, "嗯"), (1, "我同意你的看法")], segs
assert "".join(s for _k, s in segs) == text

print("qwen word-diarize smoke test PASSED")
