#!/bin/zsh
# Installs one launchd agent that runs the local FreeFlow backend at login and restarts it if it crashes:
#   com.freeflow.local  -> http://127.0.0.1:8081/v1 (Orukeet speech-to-text + on-demand Gemma cleanup)
# Models load when a dictation starts and unload after IDLE_UNLOAD_SECONDS idle (default 600, 0 = always warm).
#   IDLE_UNLOAD_SECONDS=900 ./install-autostart.sh     # change the timeout
#   ./install-autostart.sh --uninstall
set -euo pipefail
D="${0:A:h}"
LA="$HOME/Library/LaunchAgents"
UV="$(command -v uv)"
LABEL=com.freeflow.local
IDLE="${IDLE_UNLOAD_SECONDS:-600}"
LLM_MODEL="${LLM_MODEL:-mlx-community/gemma-4-e4b-it-4bit}"

for label in com.freeflow.asr com.freeflow.llm "$LABEL"; do  # also removes the older two-agent layout
  launchctl bootout "gui/$UID/$label" 2>/dev/null || true
  rm -f "$LA/$label.plist"
  # bootout returns before the job is gone; bootstrapping too early fails with "5: Input/output error"
  for _ in {1..50}; do launchctl print "gui/$UID/$label" >/dev/null 2>&1 || break; sleep 0.2; done
done
[[ "${1:-}" == "--uninstall" ]] && { echo "Removed FreeFlow local agent."; exit 0; }

mkdir -p "$D/logs" "$LA"
cat > "$LA/$LABEL.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$UV</string><string>run</string><string>--frozen</string><string>uvicorn</string>
    <string>server:app</string><string>--host</string><string>127.0.0.1</string><string>--port</string><string>8081</string>
  </array>
  <key>WorkingDirectory</key><string>$D</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin</string>
    <key>HF_HUB_OFFLINE</key><string>1</string>
    <key>IDLE_UNLOAD_SECONDS</key><string>$IDLE</string>
    <key>LLM_MODEL</key><string>$LLM_MODEL</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$D/logs/server.log</string>
  <key>StandardErrorPath</key><string>$D/logs/server.log</string>
</dict></plist>
EOF
plutil -lint -s "$LA/$LABEL.plist"
for attempt in {1..5}; do
  launchctl bootstrap "gui/$UID" "$LA/$LABEL.plist" 2>/dev/null && break
  (( attempt == 5 )) && { launchctl bootstrap "gui/$UID" "$LA/$LABEL.plist"; exit 1; }  # show the error
  sleep 1
done
echo "Installed (idle unload after ${IDLE}s). Logs: $D/logs/server.log, $D/logs/llm.log"
