"""Locating the external binary (audiotee).

ASR/VAD model downloads are handled by mlx-audio's load.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_audiotee(explicit: str | None) -> str:
    """Locate the audiotee binary: explicit path > project ./bin > PATH."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    candidates.append(str(_PROJECT_ROOT / "bin" / "audiotee"))
    found = shutil.which("audiotee")
    if found:
        candidates.append(found)

    for c in candidates:
        # Must be an executable regular file: a directory or a file without the execute
        # bit left to Popen only yields a cryptic error
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c

    raise FileNotFoundError(
        "audiotee binary not found. System audio capture needs it; build it first:\n"
        "    bash scripts/build_audiotee.sh\n"
        "or pass an existing binary path with --audiotee."
    )
