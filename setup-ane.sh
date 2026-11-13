#!/bin/zsh
# Set up Orukeet on the Apple Neural Engine (Core ML). Run via `freeflowctl ane on`; safe to re-run.
#   1. download Oruk AI's Core ML bundle (6-bit weights, ~470 MB) and verify its published SHA-256
#   2. build the orukeet-ane helper from ane/orukeet_ane.swift (needs Xcode Command Line Tools), unless a release
#      already shipped it prebuilt in bin/
#   3. add JointDecisionv3Logits: Oruk AI's joint model with its existing scores as an extra output, for vocabulary
#      boosting. Made here from the verified download plus ane/JointDecisionv3Logits.mlmodel (the 9 KB graph
#      description; no weights): the weight files are Oruk AI's, unchanged. See tools/make_joint_logits.py.
#   4. compile the model for this Mac's Neural Engine (~15 s once; cached by macOS afterwards)
set -euo pipefail
D="${0:A:h}"
REPO=oruk/orukeet
REVISION=coreml-taptalk-preview-20260915
ARCHIVE=coreml/orukeet-r3-coreml-greedy.zip
SHA256=beccdc6f18c4b10527a764f6e3ab12e3e11b969220c0cee175b3bb7eaa94290e  # from Oruk AI's coreml/README.md
BUNDLE="$D/coreml/orukeet-r3-coreml-greedy"
LOGITS_SHA256=3f6b402468c46ac13d12b718a98376017e8d6c68ec557b231c56286ed4dd1562  # ane/JointDecisionv3Logits.mlmodel

if [[ ! -x "$D/bin/orukeet-ane" ]] || [[ "$D/ane/orukeet_ane.swift" -nt "$D/bin/orukeet-ane" ]]; then
  command -v swiftc >/dev/null || { echo "Needs Xcode Command Line Tools: xcode-select --install" >&2; exit 1; }
  BUILD_HELPER=1
fi
cd "$D"

if [[ ! -d "$BUNDLE" ]]; then
  echo "Downloading Orukeet Core ML model (~470 MB)..."
  if ! zip=$(uv run --frozen hf download "$REPO" "$ARCHIVE" --revision "$REVISION" --quiet 2>&1 | tail -1); then
    echo "$zip" >&2; exit 1
  fi
  [[ "$(shasum -a 256 "$zip" | awk '{print $1}')" == "$SHA256" ]] || { echo "Checksum mismatch for $zip" >&2; exit 1; }
  rm -rf "$D/coreml" && mkdir -p "$D/coreml"
  unzip -q "$zip" -d "$D/coreml"
fi

LOGITS="$BUNDLE/JointDecisionv3Logits.mlpackage"
if [[ ! -f "$LOGITS/Data/com.apple.CoreML/model.mlmodel" ]] || \
   [[ "$(shasum -a 256 "$LOGITS/Data/com.apple.CoreML/model.mlmodel" | awk '{print $1}')" != "$LOGITS_SHA256" ]]; then
  [[ "$(shasum -a 256 "$D/ane/JointDecisionv3Logits.mlmodel" | awk '{print $1}')" == "$LOGITS_SHA256" ]] \
    || { echo "Checksum mismatch for ane/JointDecisionv3Logits.mlmodel" >&2; exit 1; }
  rm -rf "$LOGITS.tmp" && cp -R "$BUNDLE/JointDecisionv3.mlpackage" "$LOGITS.tmp"
  cp "$D/ane/JointDecisionv3Logits.mlmodel" "$LOGITS.tmp/Data/com.apple.CoreML/model.mlmodel"
  rm -rf "$LOGITS" && mv "$LOGITS.tmp" "$LOGITS"
fi

if [[ -n "${BUILD_HELPER:-}" ]]; then
  echo "Building the Neural Engine helper..."
  mkdir -p "$D/bin"
  swiftc -O -o "$D/bin/orukeet-ane" "$D/ane/orukeet_ane.swift"
fi

echo "Compiling the model for this Mac's Neural Engine (about 15 s the first time)..."
rm -rf "$D/coreml/compiled"  # compiled models are tied to this macOS version and chip
"$D/bin/orukeet-ane" "$BUNDLE" "$D/coreml/compiled" --warmup >/dev/null
echo "Neural Engine speech-to-text is ready."
