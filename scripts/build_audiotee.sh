#!/usr/bin/env bash
# Build audiotee (system audio capture, Core Audio process tap) into the project's ./bin/audiotee.
# Only needed for the system / both audio sources; not used for mic-only. Requires macOS 14.2+ and Swift 5.9+.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$HERE/.vendor/audiotee"
BIN_DIR="$HERE/bin"
# Pin to a verified commit (supply-chain: do not build a third-party repo's HEAD directly);
# to upgrade, set the AUDIOTEE_REF environment variable and re-verify the system source
AUDIOTEE_REF="${AUDIOTEE_REF:-56ac954369a09318e46b88a6eec33c2d2b0d32a3}"

mkdir -p "$BIN_DIR"

if [ ! -d "$VENDOR/.git" ]; then
    echo "==> Cloning audiotee source"
    git clone https://github.com/makeusabrew/audiotee.git "$VENDOR"
fi
echo "==> Checking out pinned revision $AUDIOTEE_REF"
git -C "$VENDOR" rev-parse --quiet --verify "$AUDIOTEE_REF^{commit}" >/dev/null \
    || git -C "$VENDOR" fetch origin
git -C "$VENDOR" checkout --quiet "$AUDIOTEE_REF"

echo "==> swift build -c release"
cd "$VENDOR"
swift build -c release

BIN="$(swift build -c release --show-bin-path)/audiotee"
cp "$BIN" "$BIN_DIR/audiotee"
echo "==> Done: $BIN_DIR/audiotee"
echo
echo "On first use of the system source, macOS may prompt for 'Screen & System Audio"
echo "Recording' permission for your terminal app. If no prompt appears, grant it"
echo "manually: System Settings > Privacy & Security > Screen & System Audio Recording."
echo "On macOS 15+ use the 'System Audio Recording Only' sub-section (NOT the top one),"
echo "then fully quit and restart the terminal. See README."
