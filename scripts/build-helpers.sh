#!/bin/zsh
# Build the two Swift helpers into bin/ (needs the Xcode Command Line Tools: xcode-select --install).
#   bin/orukeet-ane   Orukeet speech-to-text on the Neural Engine (Core ML)
#   bin/screen-ocr    reads FreeFlow's context screenshot (Vision), then checks itself
# Release builds use this too, so a downloaded release needs no compiler.
set -euo pipefail
cd "${0:A:h}/.."
TARGET=arm64-apple-macos14.0  # oldest macOS the binaries run on
mkdir -p bin
swiftc -O -target "$TARGET" -o bin/orukeet-ane ane/orukeet_ane.swift
swiftc -O -target "$TARGET" -o bin/screen-ocr ocr/screen_ocr.swift
swiftc -O -target "$TARGET" -o bin/edit-watch learn/edit_watch.swift
bin/screen-ocr --selftest >/dev/null || { echo "screen-ocr failed its self-test" >&2; exit 1; }
echo "Built bin/orukeet-ane, bin/screen-ocr and bin/edit-watch for $TARGET."
