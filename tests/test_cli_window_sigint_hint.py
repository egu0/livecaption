from __future__ import annotations

import io
import signal
import unittest
from unittest.mock import patch

from livecaption import cli_window


class CliWindowSigintHintTests(unittest.TestCase):
    def test_sigint_handler_prints_escape_close_hint(self) -> None:
        stderr = io.StringIO()

        with (
            patch("livecaption.cli_window.signal.signal") as install_signal,
            patch("livecaption.cli_window.sys.stderr", stderr),
        ):
            cli_window.install_sigint_hint_handler()
            signum, handler = install_signal.call_args.args
            self.assertEqual(signum, signal.SIGINT)

            handler(signal.SIGINT, None)

        self.assertIn("Ctrl+C does not close the caption window", stderr.getvalue())
        self.assertIn("Press Esc twice to close", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
