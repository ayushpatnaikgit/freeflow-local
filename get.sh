#!/bin/sh
# One-line installer for freeflow-local: private, offline dictation on Apple Silicon.
#
#   curl -fsSL https://raw.githubusercontent.com/ayushpatnaikgit/freeflow-local/main/get.sh | sh
#
# 1. Installs FreeFlow (the dictation app) if it isn't installed: downloads the official FreeFlow.dmg from
#    github.com/zachlatta/freeflow and installs it only if it is signed by FreeFlow's developer and notarized by Apple.
# 2. Installs uv (the Python package manager) if missing.
# 3. Downloads the latest freeflow-local release (source plus prebuilt helpers, so no Xcode tools are needed), checks
#    its SHA-256 and runs its install.sh, which starts the backend at login and points FreeFlow at it.
#
# Options: --no-freeflow (don't install FreeFlow); anything else is passed to install.sh, e.g.
#   curl -fsSL .../get.sh | sh -s -- --no-configure
# Environment: FREEFLOW_LOCAL_RELEASE (release download URL), FREEFLOW_DMG_URL, FREEFLOW_APPS_DIR (where to put
# FreeFlow.app; default /Applications, or ~/Applications if that isn't writable), plus install.sh's
# IDLE_UNLOAD_SECONDS, LLM_MODEL and FREEFLOW_LOCAL_DIR.
set -eu

REPO=ayushpatnaikgit/freeflow-local
BASE="${FREEFLOW_LOCAL_RELEASE:-https://github.com/$REPO/releases/latest/download}"
NAME=freeflow-local-macos-arm64.tar.gz
FREEFLOW_DMG="${FREEFLOW_DMG_URL:-https://github.com/zachlatta/freeflow/releases/latest/download/FreeFlow.dmg}"
FREEFLOW_ID=com.zachlatta.freeflow  # FreeFlow's bundle identifier
FREEFLOW_TEAM=P6PV2R9443            # Apple Developer team that signs FreeFlow releases
APPS="${FREEFLOW_APPS_DIR:-}"

say() { printf '%s\n' "$*"; }
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

INSTALL_FREEFLOW=1
for arg in "$@"; do
  shift
  case "$arg" in
    --no-freeflow) INSTALL_FREEFLOW=0 ;;
    *) set -- "$@" "$arg" ;;
  esac
done

[ "$(uname -s)" = Darwin ] || fail "freeflow-local runs on macOS only."
[ "$(uname -m)" = arm64 ] || fail "freeflow-local needs an Apple Silicon Mac (M1 or later)."
major=$(sw_vers -productVersion | cut -d. -f1)
[ "$major" -ge 14 ] || fail "freeflow-local needs macOS 14 (Sonoma) or later."

WORK=$(mktemp -d "${TMPDIR:-/tmp}/freeflow-local.XXXXXX")
MOUNT="$WORK/mnt"
cleanup() {
  if [ -d "$MOUNT" ]; then hdiutil detach -quiet "$MOUNT" 2>/dev/null || true; fi
  rm -rf "${WORK:?}"
}
trap cleanup EXIT INT TERM

have_freeflow() {
  if [ -n "$APPS" ]; then
    [ -d "$APPS/FreeFlow.app" ]
  else
    [ -d /Applications/FreeFlow.app ] || [ -d "$HOME/Applications/FreeFlow.app" ]
  fi
}

install_freeflow() {
  say "Installing FreeFlow (the dictation app) from github.com/zachlatta/freeflow..."
  curl -fsSL "$FREEFLOW_DMG" -o "$WORK/FreeFlow.dmg" || fail "could not download FreeFlow from $FREEFLOW_DMG"
  mkdir -p "$MOUNT"
  hdiutil attach -nobrowse -readonly -noautoopen -quiet -mountpoint "$MOUNT" "$WORK/FreeFlow.dmg" \
    || fail "could not open the FreeFlow disk image."
  app="$MOUNT/FreeFlow.app"
  [ -d "$app" ] || fail "the FreeFlow disk image does not contain FreeFlow.app."
  # Only install the genuine app: intact signature, FreeFlow's identifier and team, notarized by Apple.
  codesign --verify --deep --strict "$app" 2>/dev/null || fail "FreeFlow's code signature is not valid; not installing it."
  sig=$(codesign -dv "$app" 2>&1)
  printf '%s\n' "$sig" | grep -qx "Identifier=$FREEFLOW_ID" \
    && printf '%s\n' "$sig" | grep -qx "TeamIdentifier=$FREEFLOW_TEAM" \
    || fail "FreeFlow.dmg is not signed by FreeFlow's developer; not installing it. Install FreeFlow yourself from
       https://github.com/zachlatta/freeflow, then run this again."
  spctl --assess --type execute "$app" 2>/dev/null || fail "FreeFlow is not notarized by Apple; not installing it."
  dest="$APPS"
  if [ -z "$dest" ]; then
    if [ -w /Applications ]; then dest=/Applications; else dest="$HOME/Applications"; fi
  fi
  mkdir -p "$dest"
  ditto "$app" "$dest/FreeFlow.app" || fail "could not copy FreeFlow to $dest."
  hdiutil detach -quiet "$MOUNT" 2>/dev/null || true
  rmdir "$MOUNT" 2>/dev/null || true
  say "FreeFlow installed in $dest."
}

if [ "$INSTALL_FREEFLOW" = 1 ]; then
  if have_freeflow; then say "FreeFlow is already installed."; else install_freeflow; fi
fi

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

say "Downloading freeflow-local..."
curl -fsSL "$BASE/$NAME" -o "$WORK/$NAME" || fail "download failed: $BASE/$NAME"
curl -fsSL "$BASE/$NAME.sha256" -o "$WORK/$NAME.sha256" || fail "checksum download failed"
expected=$(cut -d' ' -f1 "$WORK/$NAME.sha256")
actual=$(shasum -a 256 "$WORK/$NAME" | cut -d' ' -f1)
[ -n "$expected" ] && [ "$expected" = "$actual" ] || fail "checksum mismatch for $NAME; not installing."

tar -xzf "$WORK/$NAME" -C "$WORK"
[ -x "$WORK/freeflow-local/install.sh" ] || fail "the download does not contain install.sh."
zsh "$WORK/freeflow-local/install.sh" "$@"
