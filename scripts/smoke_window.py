"""Smoke tests for the Swift caption window: SwiftCaptionWindow JSON format + entry-point sanity."""
from __future__ import annotations

import ast
import json
import queue
import sys
import threading
from pathlib import Path

# Allow running from repo root: uv run python scripts/smoke_window.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livecaption.swift_window import SwiftCaptionWindow


def test_partial_json_format():
    """partial() sends the correct JSON-line structure."""
    win = SwiftCaptionWindow.__new__(SwiftCaptionWindow)
    win._q = queue.Queue()
    win._stop_event = threading.Event()
    win._proc = None  # no real subprocess

    win.partial("me", "hello world", None)
    line = win._q.get(timeout=1)
    obj = json.loads(line.strip())
    assert obj == {"type": "partial", "text": "hello world"}, f"Unexpected: {obj}"


def test_final_json_format():
    """final() flattens segments and sends the correct JSON."""
    win = SwiftCaptionWindow.__new__(SwiftCaptionWindow)
    win._q = queue.Queue()
    win._stop_event = threading.Event()
    win._proc = None

    win.final("me", [(None, "hello final", None)], None)
    line = win._q.get(timeout=1)
    obj = json.loads(line.strip())
    assert obj == {"type": "final", "text": "hello final"}, f"Unexpected: {obj}"


def test_final_multi_segment_join():
    """Multiple segments are joined with double-space."""
    win = SwiftCaptionWindow.__new__(SwiftCaptionWindow)
    win._q = queue.Queue()
    win._stop_event = threading.Event()
    win._proc = None

    win.final("me", [(0, "speaker one", None), (1, "speaker two", None)], None)
    line = win._q.get(timeout=1)
    obj = json.loads(line.strip())
    assert obj == {"type": "final", "text": "speaker one  speaker two"}, f"Unexpected: {obj}"


def test_status_json_format():
    """set_status() sends the correct JSON."""
    win = SwiftCaptionWindow.__new__(SwiftCaptionWindow)
    win._q = queue.Queue()
    win._stop_event = threading.Event()
    win._proc = None

    win.set_status("● Listening")
    line = win._q.get(timeout=1)
    obj = json.loads(line.strip())
    assert obj == {"type": "status", "message": "● Listening"}, f"Unexpected: {obj}"


def test_thread_safety():
    """Concurrent sends from multiple threads don't corrupt JSON lines."""
    win = SwiftCaptionWindow.__new__(SwiftCaptionWindow)
    win._q = queue.Queue()
    win._stop_event = threading.Event()
    win._proc = FakeProc()
    win._writer = None  # don't start writer — just check enqueue ordering

    n = 50
    barrier = threading.Barrier(2)

    def producer(prefix: str) -> None:
        barrier.wait()
        for i in range(n):
            win.partial(prefix, f"{prefix}-{i}", None)

    t1 = threading.Thread(target=producer, args=("A",))
    t2 = threading.Thread(target=producer, args=("B",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # All lines should be drainable and valid JSON
    count = 0
    a_count = 0
    b_count = 0
    while True:
        try:
            line = win._q.get_nowait()
        except queue.Empty:
            break
        obj = json.loads(line.strip())
        assert obj["type"] == "partial"
        count += 1
        if obj["text"].startswith("A-"):
            a_count += 1
        elif obj["text"].startswith("B-"):
            b_count += 1
    assert count == 2 * n, f"Expected {2 * n} lines, got {count}"
    assert a_count == n
    assert b_count == n


class FakeProc:
    """Stub subprocess for tests that exercise the writer loop."""
    def poll(self):
        return None
    def terminate(self):
        pass
    def wait(self, timeout: float | None = None):
        pass
    def kill(self):
        pass


def test_writer_stops_on_broken_pipe():
    """Writer loop exits cleanly when the pipe breaks."""
    win = SwiftCaptionWindow.__new__(SwiftCaptionWindow)
    win._q = queue.Queue()
    win._stop_event = threading.Event()
    win._proc = FakeProc()

    # Put one line and immediately break the pipe simulation
    win._send({"type": "partial", "text": "test"})

    # Simulate: proc.poll() returns non-None (process died)
    win._proc.poll = lambda: 1  # exit code

    # Writer loop should exit without error
    win._write_loop()  # should not hang or raise
    assert True  # reached without exception


def test_window_entry_point_func_exists():
    """cli_window.main is importable and callable (Typer function)."""
    from livecaption.cli_window import main as window_main
    assert callable(window_main), "main should be callable"


def test_window_entry_point_no_translate_import():
    """The window entry point must not import or construct Translator."""
    cli_path = (
        Path(__file__).resolve().parent.parent / "livecaption" / "cli_window.py"
    )
    with open(cli_path) as f:
        src = f.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (
            node.module == "livecaption.translate"
            or (node.module == "translate" and node.level == 1)
        ):
            raise AssertionError("cli_window.py must not import translate")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "translate" in alias.name:
                    raise AssertionError("cli_window.py must not import translate")


def test_window_entry_point_no_diarize():
    """Window mode always passes diarize=False to build_recognizer."""
    cli_path = (
        Path(__file__).resolve().parent.parent / "livecaption" / "cli_window.py"
    )
    with open(cli_path) as f:
        src = f.read()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in getattr(node, "keywords", []):
                if kw.arg == "diarize":
                    val = kw.value
                    if isinstance(val, ast.Constant):
                        if val.value is not False:
                            raise AssertionError(
                                f"diarize must be False, got {val.value}"
                            )
                        found = True
                    elif isinstance(val, ast.Name) and val.id == "False":
                        found = True
    if not found:
        raise AssertionError("build_recognizer diarize= not found in cli_window.py")


def test_window_entry_point_no_mic_source():
    """Window mode must not import or construct MicSource."""
    cli_path = (
        Path(__file__).resolve().parent.parent / "livecaption" / "cli_window.py"
    )
    with open(cli_path) as f:
        src = f.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "MicSource" in [
            a.name for a in node.names
        ]:
            raise AssertionError("cli_window.py must not import MicSource")


if __name__ == "__main__":
    tests = [
        test_partial_json_format,
        test_final_json_format,
        test_final_multi_segment_join,
        test_status_json_format,
        test_thread_safety,
        test_writer_stops_on_broken_pipe,
        test_window_entry_point_func_exists,
        test_window_entry_point_no_translate_import,
        test_window_entry_point_no_diarize,
        test_window_entry_point_no_mic_source,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    if passed < len(tests):
        sys.exit(1)
