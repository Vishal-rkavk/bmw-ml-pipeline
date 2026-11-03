#!/usr/bin/env bash
# Stop the monitoring stack started by monitoring/up.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=monitoring/runtime
for name in grafana alertmanager prometheus pushgateway; do
  pidfile="$RUN/$name.pid"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    kill "$(cat "$pidfile")"
    rm -f "$pidfile"
    echo "[monitoring] $name stopped"
  else
    rm -f "$pidfile"
    echo "[monitoring] $name not running"
  fi
done
