"""Pure-logic smoke test for _inline_diff (no model dependency, runs in seconds).

Verifies the two-pass correction's inline diff spans: an unchanged segment is None; same+add
reassembled equals the new text; del is the corrected-away old word; a trailing deletion anchors
to the last segment; a cross-segment replace slices correctly.
    uv run python scripts/smoke_diff.py
"""

from livecaption.asr import _inline_diff


def words(spans: list[tuple[str, str]], kinds: set[str]) -> str:
    return " ".join(w for k, w in spans if k in kinds)


# 1. No correction at all → every segment is None
assert _inline_diff("hello world", ["hello world"]) == [None]
assert _inline_diff("a b c d", ["a b", "c d"]) == [None, None]

# 2. Single-segment replace: same+add reassembled == new text, del == corrected-away word
spans = _inline_diff("i red a book", ["i read a book"])[0]
assert spans is not None
assert words(spans, {"same", "add"}) == "i read a book"
assert words(spans, {"del"}) == "red"

# 3. Multiple segments (diarize split): replace + insert distributed across different
#    segments, each segment's reassembly equals the segment text
old = "the quick brown fox jump over lazy dog"
segs = ["the quick brown fox jumps", "over the lazy dog"]
out = _inline_diff(old, segs)
for seg, sp in zip(segs, out, strict=True):
    if sp is not None:
        assert words(sp, {"same", "add"}) == seg, (seg, sp)
assert words(out[0], {"del"}) == "jump"

# 4. Trailing deletion anchors to the last segment (the scenario where two-pass corrects
#    away hallucinated trailing words)
out = _inline_diff("a b c d e", ["a b", "c"])
flat_del = " ".join(w for sp in out if sp for k, w in sp if k == "del")
assert "d" in flat_del and "e" in flat_del, out

# 5. When only one segment changes, the unchanged segment stays None
out = _inline_diff("one two three four", ["one two", "three five"])
assert out[0] is None and out[1] is not None, out
assert words(out[1], {"del"}) == "four"

# 6. The whole sentence is rewritten
spans = _inline_diff("completely wrong", ["totally different text"])[0]
assert spans is not None
assert words(spans, {"same", "add"}) == "totally different text"
assert words(spans, {"del"}) == "completely wrong"

print("inline diff smoke test PASSED")
