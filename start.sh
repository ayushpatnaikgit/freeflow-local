#!/bin/zsh
# Run the backend in the foreground (Ctrl-C stops it). Use `freeflowctl install` to run it at login instead.
set -euo pipefail
cd "${0:A:h}"
export HF_HUB_OFFLINE=1 IDLE_UNLOAD_SECONDS="${IDLE_UNLOAD_SECONDS:-600}"
exec uv run --frozen uvicorn server:app --host 127.0.0.1 --port 8081
