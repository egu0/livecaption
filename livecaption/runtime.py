"""Process-wide MLX serialization lock.

MLX offers no official guarantee for concurrent multi-threaded evaluation (mlx-lm's own
server also processes requests serially), and this hazard is process-wide: both the ASR
thread (VAD/encoder/RNNT/diar) and the translation thread (mlx-lm) trigger mx evaluation.
So the lock lives in this neutral module for whoever needs it to import — asr holds the
lock to run a whole segment of computation (milliseconds), translation acquires it per
decode step (see translate._translate), ensuring that at any moment only one user thread
is calling the mx API, while not letting a single translation freeze partials for long.
"""

from __future__ import annotations

import threading

MLX_LOCK = threading.Lock()
