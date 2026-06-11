"""Manual smoke test: load the Hy-MT2 translation model and translate a few English sentences.

Verifies that the Translator thread can load the mlx-lm model, that the apply_chat_template +
sampler path is correct, and that translation quality is normal.
    uv run python scripts/smoke_translate.py
"""

import threading
from datetime import datetime

from livecaption.config import DEFAULT_MT_MODEL, DEFAULT_TARGET_LANG
from livecaption.translate import Translator

SAMPLES = [
    "after early nightfall the yellow lamps would light up here and there",
    "let's get started with today's meeting agenda and review last week's action items",
]

results: list[tuple[str, str]] = []
done = threading.Event()
ready = threading.Event()


def on_translation(label: str, src: str, zh: str, started_at: datetime) -> None:
    results.append((src, zh))
    print(f"EN: {src}\nZH: {zh}\n")
    if len(results) == len(SAMPLES):
        done.set()


t = Translator(DEFAULT_MT_MODEL, DEFAULT_TARGET_LANG, on_translation, on_ready=ready.set)
t.start()
print(f"loading {DEFAULT_MT_MODEL} …")
ready.wait()
print("model ready, translating…\n")
for i, s in enumerate(SAMPLES):
    t.submit(f"s{i}", s, datetime.now())

ok = done.wait(timeout=180)
t.stop()
t.join(timeout=5)
print("translate smoke test PASSED" if ok else "translate smoke test FAILED (timeout)")
