#!/usr/bin/env bash
# Build the native Swift caption window binary into the project's ./bin/livecaption-window.
# First-party source under native/CaptionWindow/; requires macOS 14.2+ and Swift 5.9+.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_DIR="$HERE/native/CaptionWindow"
BIN_DIR="$HERE/bin"

mkdir -p "$BIN_DIR"

echo "==> Swift build -c release"
cd "$SWIFT_DIR"
swift build -c release

BIN="$(swift build -c release --show-bin-path)/livecaption-window"
cp "$BIN" "$BIN_DIR/livecaption-window"
echo "==> Done: $BIN_DIR/livecaption-window"
