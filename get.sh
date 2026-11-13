#!/bin/sh
# One-line installer for freeflow-local: private, offline dictation on Apple Silicon.
#
#   curl -fsSL https://raw.githubusercontent.com/ayushpatnaikgit/freeflow-local/main/get.sh | sh
#
# 1. Installs the dictation app if it isn't installed:
#    - FreeFlow (default): the official FreeFlow.dmg from github.com/zachlatta/freeflow, installed only if it is
#      signed by FreeFlow's developer and notarized by Apple;
#    - TellWell (--app tellwell): TellWell.dmg from github.com/ayushpatnaikgit/tellwell, installed only if it matches
#      the SHA-256 published with the release and carries TellWell's bundle identifier with an intact signature.
#      (TellWell is not yet signed with an Apple Developer ID; files downloaded by curl aren't quarantined, so
#      macOS opens it without the "could not verify" prompt.)
# 2. Installs uv (the Python package manager) if missing.
# 3. Downloads the latest freeflow-local release (source plus prebuilt helpers, so no Xcode tools are needed), checks
#    its SHA-256 and runs its install.sh, which starts the backend at login and points FreeFlow at it.
#
# Options: --app freeflow|tellwell (default freeflow), --no-freeflow (don't install an app); anything else is passed
# to install.sh, e.g.
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
TELLWELL_BASE="${TELLWELL_RELEASE:-https://github.com/ayushpatnaikgit/tellwell/releases/latest/download}"
TELLWELL_ID=io.github.ayushpatnaikgit.tellwell
APPS="${FREEFLOW_APPS_DIR:-}"

say() { printf '%s\n' "$*"; }
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

INSTALL_FREEFLOW=1
APP=FreeFlow
NEXT_IS_APP=0
for arg in "$@"; do
  shift
  if [ "$NEXT_IS_APP" = 1 ]; then
    NEXT_IS_APP=0
    case "$arg" in
      freeflow|FreeFlow) APP=FreeFlow ;;
      tellwell|TellWell) APP=TellWell ;;
      *) fail "unknown app '$arg' (use freeflow or tellwell)" ;;
    esac
    continue
  fi
  case "$arg" in
    --no-freeflow) INSTALL_FREEFLOW=0 ;;
    --app) NEXT_IS_APP=1 ;;
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

have_app() {  # have_app <Name>
  if [ -n "$APPS" ]; then
    [ -d "$APPS/$1.app" ]
  else
    [ -d "/Applications/$1.app" ] || [ -d "$HOME/Applications/$1.app" ]
  fi
}

apps_dir() {
  dest="$APPS"
  if [ -z "$dest" ]; then
    if [ -w /Applications ]; then dest=/Applications; else dest="$HOME/Applications"; fi
  fi
  mkdir -p "$dest"
  printf '%s' "$dest"
}

install_tellwell() {
  say "Installing TellWell (the dictation app) from github.com/ayushpatnaikgit/tellwell..."
  curl -fsSL "$TELLWELL_BASE/TellWell.dmg" -o "$WORK/TellWell.dmg" || fail "could not download TellWell.dmg"
  curl -fsSL "$TELLWELL_BASE/TellWell.dmg.sha256" -o "$WORK/TellWell.dmg.sha256" || fail "could not download its checksum"
  expected=$(cut -d' ' -f1 "$WORK/TellWell.dmg.sha256")
  actual=$(shasum -a 256 "$WORK/TellWell.dmg" | cut -d' ' -f1)
  [ -n "$expected" ] && [ "$expected" = "$actual" ] || fail "checksum mismatch for TellWell.dmg; not installing it."
  mkdir -p "$MOUNT"
  hdiutil attach -nobrowse -readonly -noautoopen -quiet -mountpoint "$MOUNT" "$WORK/TellWell.dmg" \
    || fail "could not open the TellWell disk image."
  app="$MOUNT/TellWell.app"
  [ -d "$app" ] || fail "the TellWell disk image does not contain TellWell.app."
  codesign --verify --deep --strict "$app" 2>/dev/null || fail "TellWell's code signature is broken; not installing it."
  codesign -dv "$app" 2>&1 | grep -qx "Identifier=$TELLWELL_ID" || fail "TellWell.app has the wrong identifier; not installing it."
  dest=$(apps_dir)
  rm -rf "$dest/TellWell.app"
  ditto "$app" "$dest/TellWell.app" || fail "could not copy TellWell to $dest."
  hdiutil detach -quiet "$MOUNT" 2>/dev/null || true
  rmdir "$MOUNT" 2>/dev/null || true
  say "TellWell installed in $dest."
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
  dest=$(apps_dir)
  ditto "$app" "$dest/FreeFlow.app" || fail "could not copy FreeFlow to $dest."
  hdiutil detach -quiet "$MOUNT" 2>/dev/null || true
  rmdir "$MOUNT" 2>/dev/null || true
  say "FreeFlow installed in $dest."
}

if [ "$INSTALL_FREEFLOW" = 1 ]; then
  if [ "$APP" = TellWell ]; then
    install_tellwell  # always the latest release, so running this again updates TellWell
  elif have_app FreeFlow; then
    say "FreeFlow is already installed."
  else
    install_freeflow
  fi
fi
if [ "$APP" = TellWell ]; then
  FREEFLOW_LOCAL_APP=tellwell
  export FREEFLOW_LOCAL_APP  # install.sh points TellWell (only) at the engine
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
