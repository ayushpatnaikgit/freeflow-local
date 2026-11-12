#!/bin/sh
# One-line installer for freeflow-local (private, offline dictation for FreeFlow on Apple Silicon):
#
#   curl -fsSL https://raw.githubusercontent.com/ayushpatnaikgit/freeflow-local/main/get.sh | sh
#
# Downloads the latest release (source plus prebuilt helpers, so no Xcode tools are needed), checks its SHA-256,
# installs uv if missing, and runs the release's install.sh. Options are passed to install.sh, e.g.
#   curl -fsSL .../get.sh | sh -s -- --no-configure
# Environment: FREEFLOW_LOCAL_RELEASE (release download URL), plus install.sh's IDLE_UNLOAD_SECONDS, LLM_MODEL and
# FREEFLOW_LOCAL_DIR.
set -eu

REPO=ayushpatnaikgit/freeflow-local
BASE="${FREEFLOW_LOCAL_RELEASE:-https://github.com/$REPO/releases/latest/download}"
NAME=freeflow-local-macos-arm64.tar.gz

say() { printf '%s\n' "$*"; }
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] || fail "freeflow-local runs on macOS only."
[ "$(uname -m)" = arm64 ] || fail "freeflow-local needs an Apple Silicon Mac (M1 or later)."
major=$(sw_vers -productVersion | cut -d. -f1)
[ "$major" -ge 14 ] || fail "freeflow-local needs macOS 14 (Sonoma) or later."

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  if command -v brew >/dev/null 2>&1; then
    say "Installing uv (Python package manager) with Homebrew..."
    brew install uv
  else
    say "Installing uv (Python package manager) from astral.sh..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  fi
fi
PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
export PATH
command -v uv >/dev/null 2>&1 || fail "uv is not available; install it from https://docs.astral.sh/uv/ and try again."

WORK=$(mktemp -d "${TMPDIR:-/tmp}/freeflow-local.XXXXXX")
trap 'rm -rf "${WORK:?}"' EXIT INT TERM

say "Downloading freeflow-local..."
curl -fsSL "$BASE/$NAME" -o "$WORK/$NAME" || fail "download failed: $BASE/$NAME"
curl -fsSL "$BASE/$NAME.sha256" -o "$WORK/$NAME.sha256" || fail "checksum download failed"
expected=$(cut -d' ' -f1 "$WORK/$NAME.sha256")
actual=$(shasum -a 256 "$WORK/$NAME" | cut -d' ' -f1)
[ -n "$expected" ] && [ "$expected" = "$actual" ] || fail "checksum mismatch for $NAME; not installing."

tar -xzf "$WORK/$NAME" -C "$WORK"
[ -x "$WORK/freeflow-local/install.sh" ] || fail "the download does not contain install.sh."
zsh "$WORK/freeflow-local/install.sh" "$@"
