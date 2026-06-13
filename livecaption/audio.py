"""Audio sources: microphone (sounddevice) and system audio (audiotee).

Both sources behave the same way: capture in the background, push float32 [-1,1]
mono chunks into self.queue, and put a SENTINEL(None) when the stream ends. The
downstream ASR thread consumes via queue.get().
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import select
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from .config import SAMPLE_RATE, SYSTEM_AUDIO_STALL_SEC

SENTINEL = None  # putting this on the queue signals the audio stream has ended


class AudioSource(ABC):
    def __init__(self, label: str):
        self.label = label
        # maxsize caps memory; when full, drop the oldest live frame rather than block
        # the capture thread
        self.queue: queue.Queue = queue.Queue(maxsize=200)
        self._stop = threading.Event()

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def _offer(self, samples: np.ndarray) -> None:
        """Enqueue a live frame: when full, drop the oldest frame, never block the capture side.

        Assumes a single producer (only one thread per source calls _offer): the
        consumer only gets, never puts, so the slot freed by get won't be stolen by
        anyone else, and the loop is guaranteed to succeed within a round or two.
        """
        while True:
            try:
                self.queue.put_nowait(samples)
                return
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self.queue.get_nowait()

    def _put_sentinel(self) -> None:
        """Enqueue SENTINEL, dropping the oldest frame to make room if needed.

        We can't use a blocking put: if the consumer (AsrWorker) has already died
        unexpectedly, a blocking put on a full queue would make stop() hang forever
        (which then hangs the cli cleanup path, unrecoverable even with Ctrl-C).
        """
        while True:
            try:
                self.queue.put_nowait(SENTINEL)
                return
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self.queue.get_nowait()


class MicSource(AudioSource):
    """Microphone. The sounddevice callback runs on the PortAudio audio thread and only enqueues."""

    def __init__(self, label: str = "me", device: int | str | None = None, block_ms: int = 100):
        super().__init__(label)
        self.device = device
        self.blocksize = int(SAMPLE_RATE * block_ms / 1000)
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if self._stop.is_set():
                return
            self._offer(indata[:, 0].copy())  # (frames, 1) float32 -> 1d

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._put_sentinel()


class FileSource(AudioSource):
    """Audio file source (for end-to-end tests / replaying meeting recordings):
    decode to 16k mono, then feed the queue in chunks.

    Unlike the live sources: use a blocking put instead of dropping frames — a file
    has no real-time constraint, so dropping frames would just lose content.
    Endpointing measures silence by VAD frame count, not wall clock, so feeding
    faster than real time doesn't affect sentence segmentation.
    Decoding uses mlx-audio's load_audio (miniaudio), which supports wav/mp3/m4a etc.
    and resamples automatically.
    """

    def __init__(self, path: str, label: str = "file", block_ms: int = 100):
        super().__init__(label)
        self.path = path
        self.blocksize = int(SAMPLE_RATE * block_ms / 1000)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._feed, daemon=True, name=f"file-{self.label}"
        )
        self._thread.start()

    def _feed(self) -> None:
        from mlx_audio.stt.utils import load_audio

        samples = np.array(load_audio(self.path, SAMPLE_RATE), dtype=np.float32)
        for i in range(0, len(samples), self.blocksize):
            if self._stop.is_set():
                break
            chunk = samples[i : i + self.blocksize]
            while not self._stop.is_set():
                try:
                    self.queue.put(chunk, timeout=0.2)
                    break
                except queue.Full:
                    continue
        self._put_sentinel()

    def stop(self) -> None:
        self._stop.set()  # SENTINEL is put by the _feed thread when it exits


class SystemAudioSource(AudioSource):
    """System audio (the meeting's speaker output), via audiotee (a Core Audio process tap).

    audiotee protocol: stdout = raw PCM; stderr = one NDJSON record per line.
    With --sample-rate 16000, the output is fixed at 16-bit signed little-endian mono.

    A supervisor thread restarts audiotee when the stream ends or stalls. A healthy tap
    delivers PCM continuously (zeros during silence); receiving nothing at all for
    SYSTEM_AUDIO_STALL_SEC means the tap died -- observed when the default output device
    changes (audio keeps playing on the new device, the tap stays on the old one whose
    IO stops, and a plain blocking read would starve the whole pipeline forever with no
    diagnostic). Respawning re-taps the current default device, so captions resume.
    """

    def __init__(
        self,
        audiotee_path: str,
        label: str = "them",
        include_pids: list[int] | None = None,
    ):
        super().__init__(label)
        self.audiotee_path = audiotee_path
        self.include_pids = include_pids or []
        self._proc: subprocess.Popen | None = None
        self._zero_warned = False  # permission warning, once per run (not per respawn)

    def start(self) -> None:
        # the first spawn raises on immediate failure (wrong path / instant crash), so
        # the CLI can report it instead of starting a silently dataless track
        self._spawn()
        threading.Thread(
            target=self._supervise, daemon=True, name=f"audiotee-{self.label}"
        ).start()

    def _spawn(self) -> None:
        cmd = [self.audiotee_path, "--sample-rate", str(SAMPLE_RATE)]
        for pid in self.include_pids:
            cmd += ["--include-processes", str(pid)]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        # a wrong audiotee path / a crash makes it exit immediately; surface that early
        # instead of going silently dataless
        time.sleep(0.3)
        if self._proc.poll() is not None:
            err = (self._proc.stderr.read() or b"").decode("utf-8", "replace")[:500]
            raise RuntimeError(
                f"audiotee failed to start (exit {self._proc.returncode}): {err.strip()}"
            )
        threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True).start()

    def _supervise(self) -> None:
        """Pump one audiotee process at a time; on EOF/stall, kill it and respawn."""
        failures = 0
        while not self._stop.is_set():
            reason = self._pump(self._proc)
            if self._stop.is_set():
                break
            self._kill_proc()
            print(
                f"\n[warn] system audio {reason}; restarting audiotee — if the output "
                "device changed, capture resumes on the new one.",
                file=sys.stderr,
            )
            try:
                self._spawn()
                failures = 0
            except Exception as e:  # noqa: BLE001
                failures += 1
                if failures >= 3:
                    print(
                        f"\n[warn] could not restart audiotee ({e}); this track has "
                        "stopped.",
                        file=sys.stderr,
                    )
                    break
                time.sleep(2.0)
        self._put_sentinel()

    def _pump(self, proc: subprocess.Popen) -> str:
        """Forward one audiotee process's PCM into the queue until it ends.

        Returns why it ended: "stream ended (audiotee exited)" or "stalled (no data for
        Ns)"; "stopped" when stop() was requested. Reads via select with a timeout
        instead of a plain blocking read, so a wedged tap is detected rather than
        blocking forever.
        """
        fd = proc.stdout.fileno()
        remainder = b""
        frames_seen = 0
        saw_audio = False
        last_data = time.monotonic()
        while not self._stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.5)
            if not ready:
                if time.monotonic() - last_data >= SYSTEM_AUDIO_STALL_SEC:
                    return f"stalled (no data for {SYSTEM_AUDIO_STALL_SEC:.0f}s)"
                continue
            buf = os.read(fd, 4096)
            if not buf:
                return "stream ended (audiotee exited)"
            last_data = time.monotonic()
            buf = remainder + buf
            # s16le: 2 bytes per sample, carry a half-sample to the next round
            n = len(buf) - (len(buf) % 2)
            chunk, remainder = buf[:n], buf[n:]
            if not chunk:
                continue
            pcm = np.frombuffer(chunk, dtype="<i2")
            # without recording permission the process tap silently returns exact 0;
            # several seconds of all-zero is most likely a permission issue
            if not saw_audio:
                if int(np.abs(pcm).max(initial=0)) > 30:
                    saw_audio = True
                else:
                    frames_seen += len(pcm)
                    if not self._zero_warned and frames_seen > SAMPLE_RATE * 8:
                        self._zero_warned = True
                        print(
                            "\n[warn] Captured ~8s of system audio but it is all silence. "
                            "If sound is actually playing, the terminal app almost certainly "
                            "lacks System Audio Recording permission. Grant it under System "
                            "Settings > Privacy & Security > Screen & System Audio Recording: "
                            "on macOS 15+ scroll to the 'System Audio Recording Only' "
                            "sub-section (NOT the top one) and add your terminal, then fully "
                            "quit and restart it. See README.",
                            file=sys.stderr,
                        )
            self._offer(pcm.astype(np.float32) / 32768.0)
        return "stopped"

    def _kill_proc(self) -> None:
        proc = self._proc
        if proc is None:
            return
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=2)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for raw in proc.stderr:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("message_type")
            if mtype == "metadata":
                enc = (msg.get("data") or {}).get("encoding", "")
                # we always request 16k => s16le expected. If it's unexpectedly float, warn early.
                if enc and "f32" in enc:
                    print(
                        f"[warn] audiotee emitted {enc}, but the parser assumes s16le; "
                        "audio will be garbled. Check that --sample-rate took effect.",
                        file=sys.stderr,
                    )
            elif mtype == "error":
                detail = (msg.get("data") or {}).get("message", line)
                print(f"[audiotee error] {detail}", file=sys.stderr)

    def stop(self) -> None:
        self._stop.set()
        self._kill_proc()
        self._put_sentinel()
