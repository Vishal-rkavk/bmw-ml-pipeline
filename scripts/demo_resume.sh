#!/usr/bin/env bash
# Automated proof that training survives a hard interruption.
#
# Runs a short training (throwaway checkpoint directory, real corpus),
# waits until two checkpoints are on disk, kills the process with
# SIGKILL — the uncatchable worst case — restarts it, and prints the
# evidence: the resume line, loss continuity across the kill, and the
# completed step count.
#
# With monitoring up (make monitoring-up) this demo also populates the
# dashboard loss curve under the separate "demo" run id, so it never
# overwrites the real run's series.
set -euo pipefail
cd "$(dirname "$0")/.."

DEMO=demo-artifacts
rm -rf "$DEMO"
mkdir -p "$DEMO/logs"

run_train() { # run_train <logfile>
  BMW_ARTIFACTS_DIR="$DEMO" BMW_MAX_STEPS=80 BMW_CHECKPOINT_EVERY=20 \
    BMW_LOG_EVERY=10 BMW_RUN_ID=demo make train >"$1" 2>&1
}

echo "[demo] run 1: training to step 80 (checkpoints every 20) ..."
run_train "$DEMO/logs/run1.log" &
DEMO_PID=$!

# Wait until at least two checkpoints exist (max 60s), then kill -9.
# The "|| true" matters: find exits 1 while the checkpoint directory
# does not exist yet, and with pipefail that would kill the script.
for _ in $(seq 1 30); do
  n=$(find "$DEMO/checkpoints" -name 'step-*.pt' 2>/dev/null | wc -l || true)
  if [ "$n" -ge 2 ]; then break; fi
  sleep 2
done
PID=$(pgrep -f "python3 -m pipeline[.]train" || true)
if [ -z "$PID" ]; then
  echo "[demo] training already finished before the kill — no split run."
else
  kill -9 $PID
  wait $DEMO_PID || true   # make reports the kill; that is expected
  echo "[demo] kill -9 sent. Checkpoints intact:" \
    "$(ls "$DEMO"/checkpoints/)"
fi

echo
echo "[demo] run 2: restarting — must resume from the newest checkpoint ..."
run_train "$DEMO/logs/run2.log"

echo
echo "[demo] ============ evidence ============"
echo "--- run 1: last lines before the kill ---"
tail -3 "$DEMO/logs/run1.log"
echo "--- run 2: resume + completion ---"
grep -E "resumed|done:" "$DEMO/logs/run2.log"
echo "--- run 2: loss right after the resume (continues, not restarted) ---"
grep "step .0/80" "$DEMO/logs/run2.log" | head -2
echo "[demo] ==================================="
