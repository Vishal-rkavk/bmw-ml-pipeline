#!/usr/bin/env bash
# Download the monitoring stack as pinned binaries into monitoring/bin/.
# Idempotent: already-downloaded components are skipped.
set -euo pipefail
cd "$(dirname "$0")"

PROMETHEUS_VERSION=2.55.1
PUSHGATEWAY_VERSION=1.11.1
ALERTMANAGER_VERSION=0.27.0
GRAFANA_VERSION=11.4.0

fetch() { # fetch <name> <url> <dir-inside-tarball>
  local name=$1 url=$2 inner=$3
  if [ -d "bin/$name" ]; then
    echo "[setup] $name already present (bin/$name)"
    return
  fi
  echo "[setup] downloading $name ..."
  local tmp
  tmp=$(mktemp -d)
  curl -fsSL "$url" -o "$tmp/pkg.tar.gz"
  tar -xzf "$tmp/pkg.tar.gz" -C "$tmp"
  mv "$tmp/$inner" "bin/$name"
  rm -rf "$tmp"
  echo "[setup] $name -> bin/$name"
}

mkdir -p bin
fetch prometheus \
  "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz" \
  "prometheus-${PROMETHEUS_VERSION}.linux-amd64"
fetch pushgateway \
  "https://github.com/prometheus/pushgateway/releases/download/v${PUSHGATEWAY_VERSION}/pushgateway-${PUSHGATEWAY_VERSION}.linux-amd64.tar.gz" \
  "pushgateway-${PUSHGATEWAY_VERSION}.linux-amd64"
fetch alertmanager \
  "https://github.com/prometheus/alertmanager/releases/download/v${ALERTMANAGER_VERSION}/alertmanager-${ALERTMANAGER_VERSION}.linux-amd64.tar.gz" \
  "alertmanager-${ALERTMANAGER_VERSION}.linux-amd64"
fetch grafana \
  "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz" \
  "grafana-v${GRAFANA_VERSION}"
