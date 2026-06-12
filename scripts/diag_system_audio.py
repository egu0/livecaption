"""Diagnose whether system audio capture works.

Mainly to confirm the "Screen & System Audio Recording" permission is in effect.

Play a few system alert sounds while capturing system output via audiotee, and report the
captured audio amplitude. max |amplitude| > 0 => permission OK; always 0 => permission not
granted (see the permissions section of the README).
    uv run python scripts/diag_system_audio.py
"""

import subprocess
import threading
import time

import numpy as np

from livecaption.models import resolve_audiotee

at = resolve_audiotee(None)
print("audiotee:", at)

proc = subprocess.Popen(
    [at, "--sample-rate", "16000"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
)


def play() -> None:
    time.sleep(0.5)
    for _ in range(4):
        subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"])


threading.Thread(target=play, daemon=True).start()

total = 0
maxamp = 0
t0 = time.time()
while time.time() - t0 < 5:
    buf = proc.stdout.read(4096)
    if not buf:
        break
    total += len(buf)
    a = np.frombuffer(buf[: len(buf) // 2 * 2], "<i2")
    if len(a):
        maxamp = max(maxamp, int(np.abs(a).max()))
proc.terminate()

print(f"captured ~{total / 2 / 16000:.1f}s, max |amplitude| = {maxamp}/32768")
print(
    "OK: system audio captured, permission works"
    if maxamp > 0
    else "FAIL: all silence — System Audio Recording permission likely missing. "
    "On macOS 15+ add your terminal to the 'System Audio Recording Only' sub-section "
    "(NOT the top one) under Privacy & Security > Screen & System Audio Recording. See README."
)
