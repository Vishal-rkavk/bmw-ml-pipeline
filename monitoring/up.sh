#!/usr/bin/env bash
# Start the monitoring stack as local binaries (Prometheus, Pushgateway,
# Alertmanager, Grafana). Downloads the pinned binaries on first run.
# Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=monitoring/bin
RUN=monitoring/runtime
if [ ! -x "$BIN/prometheus/prometheus" ]; then
  echo "[monitoring] binaries missing — downloading them first (~350 MB)"
  ./monitoring/setup.sh
fi
mkdir -p "$RUN/prometheus" "$RUN/alertmanager" "$RUN/grafana"

# Grafana provisioning reads these values at startup.
export PROMETHEUS_URL=http://localhost:9090
export GRAFANA_DASHBOARDS_PATH="$(pwd)/monitoring/grafana/dashboards"
export GF_PATHS_PROVISIONING="$(pwd)/monitoring/grafana/provisioning"
export GF_PATHS_DATA="$RUN/grafana"

start() { # start <name> <command...>
  local name=$1
  shift
  local pidfile="$RUN/$name.pid"
  if [ -f "$pidfile" ]; then
    local pid state
    pid=$(cat "$pidfile")
    state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')
    # A zombie (state Z) is a dead process whose parent never reaped it
    # (common in containers). Treat it as not running so we can restart.
    if [ -n "$state" ] && [ "${state#Z}" = "$state" ]; then
      echo "[monitoring] $name already running (pid $pid)"
      return
    fi
    rm -f "$pidfile"
  fi
  nohup "$@" >"$RUN/$name.log" 2>&1 &
  echo $! >"$pidfile"
  echo "[monitoring] $name started (pid $!)"
}

start pushgateway \
  "$BIN/pushgateway/pushgateway" --web.listen-address=:9091
start prometheus \
  "$BIN/prometheus/prometheus" \
  --config.file=monitoring/prometheus.yml \
  --storage.tsdb.path="$RUN/prometheus" \
  --web.listen-address=:9090
start alertmanager \
  "$BIN/alertmanager/alertmanager" \
  --config.file=monitoring/alertmanager.yml \
  --storage.path="$RUN/alertmanager" \
  --web.listen-address=:9093
start grafana \
  "$BIN/grafana/bin/grafana" server --homepath="$BIN/grafana"

echo "[monitoring] grafana      http://localhost:3000 (admin/admin)"
echo "[monitoring] prometheus   http://localhost:9090  (alerts: /alerts)"
echo "[monitoring] alertmanager http://localhost:9093"
echo "[monitoring] pushgateway  http://localhost:9091"
