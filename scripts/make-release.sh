#!/bin/zsh
# Package a release: the tracked source at HEAD plus freshly built helpers, so installing needs no compiler.
#   scripts/make-release.sh            -> dist/freeflow-local-macos-arm64.tar.gz (+ .sha256)
# The archive unpacks to freeflow-local/; get.sh downloads it and runs its install.sh.
set -euo pipefail
cd "${0:A:h}/.."
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Commit your changes first." >&2; exit 1; }
scripts/build-helpers.sh
NAME=freeflow-local-macos-arm64
STAGE=$(mktemp -d)
trap 'rm -rf "${STAGE:?}"' EXIT
git archive --format=tar --prefix=freeflow-local/ HEAD | tar -x -C "$STAGE"
mkdir -p "$STAGE/freeflow-local/bin"
cp bin/orukeet-ane bin/screen-ocr "$STAGE/freeflow-local/bin/"
git rev-parse HEAD > "$STAGE/freeflow-local/COMMIT"
mkdir -p dist
tar -C "$STAGE" -czf "dist/$NAME.tar.gz" freeflow-local
(cd dist && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "dist/$NAME.tar.gz ($(du -h "dist/$NAME.tar.gz" | cut -f1)), $(cut -d' ' -f1 "dist/$NAME.tar.gz.sha256")"
