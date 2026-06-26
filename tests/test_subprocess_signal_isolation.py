from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from livecaption.audio import SystemAudioSource
from livecaption.swift_window import SwiftCaptionWindow


class SubprocessSignalIsolationTests(unittest.TestCase):
    def test_system_audio_source_starts_audiotee_in_separate_session(self) -> None:
        proc = Mock()
        proc.poll.return_value = None
        proc.stderr = []

        source = SystemAudioSource("/tmp/audiotee")

        with (
            patch("livecaption.audio.subprocess.Popen", return_value=proc) as popen,
            patch("livecaption.audio.time.sleep"),
            patch("livecaption.audio.threading.Thread"),
        ):
            source._spawn()

        self.assertIs(popen.call_args.kwargs["start_new_session"], True)

    def test_swift_caption_window_starts_window_in_separate_session(self) -> None:
        proc = Mock()
        proc.wait.return_value = 0

        window = SwiftCaptionWindow.__new__(SwiftCaptionWindow)
        window._binary = "/tmp/livecaption-window"
        window._proc = None
        window._writer = None

        with (
            patch("livecaption.swift_window.subprocess.Popen", return_value=proc) as popen,
            patch("livecaption.swift_window.threading.Thread"),
        ):
            window.show()

        self.assertIs(popen.call_args.kwargs["start_new_session"], True)


if __name__ == "__main__":
    unittest.main()
