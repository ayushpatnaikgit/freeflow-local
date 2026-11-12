#!/bin/zsh
# Install or update the FreeFlow local backend on an Apple Silicon Mac.
#
#   ./install.sh                  install, start at login, point FreeFlow at it
#   ./install.sh --no-autostart   don't install the login agent (run ./start.sh yourself)
#   ./install.sh --no-configure   don't change FreeFlow's settings
#
# Environment: IDLE_UNLOAD_SECONDS (default 600, 0 = always warm), LLM_MODEL, FREEFLOW_LOCAL_DIR.
set -euo pipefail
SRC="${0:A:h}"
DEST="${FREEFLOW_LOCAL_DIR:-$HOME/Library/Application Support/FreeFlowLocal}"
ASR_MODEL="itsreidar/orukeet-mlx"
export LLM_MODEL="${LLM_MODEL:-mlx-community/gemma-4-e4b-it-4bit}"
AUTOSTART=1
CONFIGURE=1
for arg in "$@"; do
  case "$arg" in
    --no-autostart) AUTOSTART=0 ;;
    --no-configure) CONFIGURE=0 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

[[ "$(uname -m)" == arm64 ]] || { echo "This needs an Apple Silicon Mac (M1 or later); MLX does not run on Intel." >&2; exit 1; }
mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
(( mem_gb >= 16 )) || echo "Warning: ${mem_gb} GB of RAM. The models use about 6.5 GB while you dictate; 16 GB or more is recommended."

if ! command -v uv >/dev/null; then
  if command -v brew >/dev/null; then
    echo "Installing uv with Homebrew..."
    brew install uv
  else
    echo "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
  fi
fi

echo "Installing to: $DEST"
mkdir -p "$DEST/logs" "$DEST/ane" "$DEST/ocr" "$DEST/bin"
rm -f "$DEST/parakeet_server.py"  # renamed to server.py in 0.1.0
for f in server.py fast_tdt.py screen_words.py freeflowctl install-autostart.sh start.sh setup-ane.sh bench_fast.py \
         ane/orukeet_ane.swift ocr/screen_ocr.swift pyproject.toml uv.lock README.md LICENSE NOTICE; do
  cp "$SRC/$f" "$DEST/$f"
done
chmod +x "$DEST/freeflowctl" "$DEST/install-autostart.sh" "$DEST/start.sh" "$DEST/setup-ane.sh"

cd "$DEST"
echo "Setting up Python environment..."
uv sync --frozen --quiet --compile-bytecode
echo "Downloading models (about 6 GB the first time; later runs skip this)..."
for model in "$ASR_MODEL" "$LLM_MODEL"; do
  # Only show the downloader's output if it fails (it otherwise prints tips unrelated to this install).
  if ! out=$(uv run --frozen hf download "$model" --quiet 2>&1); then
    print -r -- "$out" >&2
    exit 1
  fi
done

# Swift helpers: a release (get.sh) ships them prebuilt in bin/; a git checkout builds them, which needs the
# Xcode Command Line Tools. Without either, screen words and Neural Engine speech-to-text are unavailable.
rm -f "$DEST/bin/screen-ocr"
if [[ -x "$SRC/bin/screen-ocr" && -x "$SRC/bin/orukeet-ane" && ! "$SRC/ocr/screen_ocr.swift" -nt "$SRC/bin/screen-ocr" ]]; then
  cp "$SRC/bin/screen-ocr" "$SRC/bin/orukeet-ane" "$DEST/bin/"
elif command -v swiftc >/dev/null; then
  echo "Building the screen-words helper..."
  swiftc -O -o "$DEST/bin/screen-ocr" "$DEST/ocr/screen_ocr.swift" || rm -f "$DEST/bin/screen-ocr"
else
  echo "Note: Xcode Command Line Tools not found (xcode-select --install), so screen words are off."
fi
if [[ -x "$DEST/bin/screen-ocr" ]] && ! "$DEST/bin/screen-ocr" --selftest >/dev/null; then
  rm -f "$DEST/bin/screen-ocr"
  echo "Warning: the screen-words helper failed its self-test; screen words are off." >&2
fi

# Neural Engine speech-to-text was switched on earlier (freeflowctl ane on): rebuild its helper for this version.
if grep -qx "ASR_BACKEND=ane" settings.env 2>/dev/null; then
  ./setup-ane.sh
fi

if (( AUTOSTART )); then
  ./install-autostart.sh
fi
if (( CONFIGURE )); then
  if [[ -d /Applications/FreeFlow.app ]]; then
    ./freeflowctl configure
  else
    echo "FreeFlow isn't installed yet. Install it from https://github.com/zachlatta/freeflow, then run:"
    echo "  \"$DEST/freeflowctl\" configure"
  fi
fi

cat <<EOF

Done. Next steps:
  1. Open FreeFlow and finish its setup: allow Microphone and Accessibility.
     If it asks for an API key, type any text (e.g. "local"). Screen Recording is optional.
  2. Hold your dictation key and speak. The first dictation after a break takes a few
     seconds while the models load; after that it's about 1 second.

Check status any time:  "$DEST/freeflowctl" status
EOF
