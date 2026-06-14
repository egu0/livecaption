"""Locating the external binary (audiotee).

ASR/VAD model downloads are handled by mlx-audio's load.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_binary(name: str, hint: str) -> str:
    """Shared 3-tier lookup for project binaries: explicit path → project ./bin → PATH."""
    candidates: list[str] = []
    if hint:
        candidates.append(hint)
    candidates.append(str(_PROJECT_ROOT / "bin" / name))
    found = shutil.which(name)
    if found:
        candidates.append(found)

    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c

    raise FileNotFoundError(
        f"{name} binary not found. Build it first:\n"
        f"    bash scripts/build_{name.replace('-','_')}.sh\n"
        f"or pass an existing binary path with --{name.replace('-','_')}."
    )


def resolve_audiotee(explicit: str | None) -> str:
    """Locate the audiotee binary: explicit path > project ./bin > PATH."""
    return _resolve_binary("audiotee", explicit or "")


def resolve_caption_window(explicit: str | None) -> str:
    """Locate the livecaption-window binary: explicit path > project ./bin > PATH."""
    return _resolve_binary("livecaption-window", explicit or "")
