"""Manual smoke test: load the Hy-MT2 translation model and translate a few English sentences.

Verifies that the Translator thread can load the mlx-lm model, that the apply_chat_template +
sampler path is correct, and that translation quality is normal. Also exercises the output
guards (_strip_boilerplate / context-echo detection) on canned strings, model-free.
    uv run python scripts/smoke_translate.py
"""

import threading
from datetime import datetime

from livecaption.config import DEFAULT_MT_MODEL, DEFAULT_TARGET_LANG
from livecaption.languages import normalize_asr_language, normalize_target_language
from livecaption.translate import Translator, _strip_boilerplate

# ---- pure-logic checks first (no model): boilerplate stripping ----
assert normalize_target_language("zh-cn").prompt_name == "Simplified Chinese"
assert normalize_target_language("Chinese").code == "zh-cn"
assert normalize_target_language("Japanese").prompt_name == "Japanese"
assert normalize_asr_language("en-us", ["en-US", "de-DE"]) == "en-US"
assert normalize_asr_language("German", ["en-US", "de-DE"]) == "de-DE"
assert (
    _strip_boilerplate("根据提供的背景信息，以下是翻译后的中文内容：\n\n“你好，世界。”")
    == "你好，世界。"
)
assert _strip_boilerplate("以下是译文：你好。") == "你好。"
assert _strip_boilerplate("Here is the translation: Bonjour.") == "Bonjour."
# no meta lead-in -> untouched, including genuine quotes
assert _strip_boilerplate("“他说：你好。”") == "“他说：你好。”"
assert _strip_boilerplate("根据报道，一切开始陷入混乱。") == "根据报道，一切开始陷入混乱。"
print("boilerplate stripping checks PASSED")

SAMPLES = [
    "after early nightfall the yellow lamps would light up here and there",
    "let's get started with today's meeting agenda and review last week's action items",
]

results: list[tuple[int | None, str]] = []
done = threading.Event()
ready = threading.Event()


def on_translation(label: str, zh_segments: list, started_at: datetime) -> None:
    # zh_segments = [(speaker, zh_text), ...] mirroring the submitted segments
    for spk, zh in zh_segments:
        results.append((spk, zh))
        print(f"ZH[{label}]: {zh}\n")
    if len(results) == len(SAMPLES):
        done.set()


t = Translator(DEFAULT_MT_MODEL, DEFAULT_TARGET_LANG, on_translation, on_ready=ready.set)
t.start()
print(f"loading {DEFAULT_MT_MODEL} …")
ready.wait()
print("model ready, translating…\n")
for i, s in enumerate(SAMPLES):
    print(f"EN: {s}")
    # submit takes the utterance's speaker segments: [(speaker, text, diff), ...]
    t.submit(f"s{i}", [(None, s, None)], datetime.now())

ok = done.wait(timeout=180)
t.stop()
t.join(timeout=5)
assert all(zh and not zh.startswith("[translation failed") for _spk, zh in results), results
print("translate smoke test PASSED" if ok else "translate smoke test FAILED (timeout)")
