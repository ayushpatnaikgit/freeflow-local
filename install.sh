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
export LLM_MODEL="${LLM_MODEL:-ayushpatnaik/S1-mini-FreeFlow-cleanup-MLX-6bit}"
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
(( mem_gb >= 8 )) || echo "Warning: ${mem_gb} GB of RAM. The models use about 3 GB while you dictate; 8 GB or more is recommended."

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
for f in server.py fast_tdt.py screen_words.py compact_prompt.py vocab_learner.py vocab_guard.py pinned-models.txt freeflowctl install-autostart.sh start.sh setup-ane.sh bench_fast.py \
         ane/orukeet_ane.swift ane/JointDecisionv3Logits.mlmodel ocr/screen_ocr.swift pyproject.toml uv.lock README.md LICENSE NOTICE; do
  cp "$SRC/$f" "$DEST/$f"
done
chmod +x "$DEST/freeflowctl" "$DEST/install-autostart.sh" "$DEST/start.sh" "$DEST/setup-ane.sh"

cd "$DEST"
echo "Setting up Python environment..."
uv sync --frozen --quiet --compile-bytecode
echo "Downloading models (about 2 GB the first time; later runs skip this)..."
for model in "$ASR_MODEL" "$LLM_MODEL"; do
  # The exact version from pinned-models.txt when the model is listed there; only show output if it fails.
  rev=$(awk -v m="$model" '$1 == m {print $2}' pinned-models.txt)
  pin=(); [[ -n "$rev" ]] && pin=(--revision "$rev")
  if ! out=$(uv run --frozen hf download "$model" "${pin[@]}" --quiet 2>&1); then
    print -r -- "$out" >&2
    exit 1
  fi
done

# Swift helpers: a release (get.sh) ships them prebuilt in bin/; a git checkout builds them, which needs the
# Xcode Command Line Tools. Without either, screen words and Neural Engine speech-to-text are unavailable.
rm -f "$DEST/bin/screen-ocr"
# A prebuilt helper is used only if it is newer than its source (a stale one silently lacks newer features).
if [[ -x "$SRC/bin/screen-ocr" && -x "$SRC/bin/orukeet-ane" && ! "$SRC/ocr/screen_ocr.swift" -nt "$SRC/bin/screen-ocr" \
      && ! "$SRC/ane/orukeet_ane.swift" -nt "$SRC/bin/orukeet-ane" ]]; then
  cp "$SRC/bin/screen-ocr" "$SRC/bin/orukeet-ane" "$DEST/bin/"
elif command -v swiftc >/dev/null; then
  echo "Building the screen-words helper..."
  swiftc -O -o "$DEST/bin/screen-ocr" "$DEST/ocr/screen_ocr.swift" || rm -f "$DEST/bin/screen-ocr"
else
  echo "Note: Xcode Command Line Tools not found (xcode-select --install), so screen words are off."
fi

# edit-watch (vocabulary learning, opt-in): macOS ties its Accessibility permission to the exact binary, so it is only
# replaced when its source changes; otherwise every reinstall would silently revoke the permission.
mkdir -p "$DEST/learn"
if [[ -x "$SRC/bin/edit-watch" && ! "$SRC/learn/edit_watch.swift" -nt "$SRC/bin/edit-watch" ]]; then
  cmp -s "$SRC/bin/edit-watch" "$DEST/bin/edit-watch" || cp "$SRC/bin/edit-watch" "$DEST/bin/edit-watch"
elif ! cmp -s "$SRC/learn/edit_watch.swift" "$DEST/learn/edit_watch.swift" || [[ ! -x "$DEST/bin/edit-watch" ]]; then
  if command -v swiftc >/dev/null; then
    echo "Building the vocabulary-learning helper..."
    swiftc -O -o "$DEST/bin/edit-watch" "$SRC/learn/edit_watch.swift" || rm -f "$DEST/bin/edit-watch"
  fi
fi
cp "$SRC/learn/edit_watch.swift" "$DEST/learn/edit_watch.swift"
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
  # FREEFLOW_LOCAL_APP (set by get.sh --app) names the app just installed; otherwise configure each installed one.
  if [[ -n "${FREEFLOW_LOCAL_APP:-}" ]]; then
    ./freeflowctl configure "$FREEFLOW_LOCAL_APP"
  elif [[ -d /Applications/FreeFlow.app || -d "$HOME/Applications/FreeFlow.app" \
        || -d /Applications/TellWell.app || -d "$HOME/Applications/TellWell.app" ]]; then
    ./freeflowctl configure
  else
    echo "No dictation app found. Install TellWell (https://github.com/ayushpatnaikgit/tellwell) or FreeFlow"
    echo "(https://github.com/zachlatta/freeflow), then run:"
    echo "  \"$DEST/freeflowctl\" configure"
  fi
fi

APP_NAME=FreeFlow
[[ "${FREEFLOW_LOCAL_APP:-}" == tellwell ]] && APP_NAME=TellWell
cat <<EOF

Done. Next steps:
  1. Open $APP_NAME and finish its setup: allow Microphone and Accessibility.
     If it asks for an API key, type any text (e.g. "local"). Screen Recording is optional.
  2. Hold your dictation key and speak. The first dictation after a break takes a second or two
     while the models load; after that, cleaned text appears about 0.2-0.4 s after you stop.

Check status any time:  "$DEST/freeflowctl" status
EOF
