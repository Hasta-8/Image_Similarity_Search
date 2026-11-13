#!/usr/bin/env bash
# ImgSimiSearch local single-origin demo runtime. Does NOT start any tunnel.
#
# One FastAPI process serves the production React build, the API routes
# (/search-image, /health, /ready) and /images/... on one local origin:
#   http://127.0.0.1:${IMGSIMI_DEMO_PORT:-8080}
#
# Usage:
#   deploy/local-demo/demo.sh start
#   deploy/local-demo/demo.sh status
#   deploy/local-demo/demo.sh stop
#
# Optional overrides (environment):
#   IMGSIMI_PYTHON               interpreter with backend dependencies (default: <repo>/.venv/bin/python)
#   IMGSIMI_DEMO_PORT            listening port (default 8080)
#   IMGSIMI_DATASET_DIR          dataset root (default: <repo>/dataset)
#   IMGSIMI_RELEASE_STORE        release store (default: <repo>/backend/artifacts)
#   IMGSIMI_DEMO_STATE_DIR       pid and log directory (default: ~/.local/state/imgsimisearch-demo)
#   IMGSIMI_MAX_UPLOAD_BYTES     upload cap (default: 16777216)
#   IMGSIMI_MAX_CONCURRENT_SEARCHES  concurrent query embeddings (default: 2)
#   IMGSIMI_SEARCH_TIMEOUT_SECONDS   per-query inference timeout (default: 60)
#
# The bind address is fixed to 127.0.0.1 so the process is reachable only from
# this host (and from a tunnel that connects locally).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
FRONTEND_BUILD="$REPO_ROOT/frontend/build"

HOST="127.0.0.1"
PORT="${IMGSIMI_DEMO_PORT:-8080}"
PYTHON="${IMGSIMI_PYTHON:-$REPO_ROOT/.venv/bin/python}"
DATASET_DIR="${IMGSIMI_DATASET_DIR:-$REPO_ROOT/dataset}"
RELEASE_STORE="${IMGSIMI_RELEASE_STORE:-$BACKEND_DIR/artifacts}"
STATE_DIR="${IMGSIMI_DEMO_STATE_DIR:-$HOME/.local/state/imgsimisearch-demo}"
PID_FILE="$STATE_DIR/demo.pid"
LOG_FILE="$STATE_DIR/server.log"
BASE_URL="http://$HOST:$PORT"

die() { echo "ERROR: $*" >&2; exit 1; }

is_port_listening() {
  (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null
}

read_pid() {
  [ -f "$PID_FILE" ] && cat "$PID_FILE" || true
}

pid_is_demo_server() {
  local pid="$1"
  [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "uvicorn main:app" || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q -- "--port $PORT" || return 1
}

http_code() {
  curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE_URL$1" || true
}

cmd_start() {
  command -v curl >/dev/null || die "curl is required"
  [ -x "$PYTHON" ] || die "Python interpreter not found: $PYTHON (set IMGSIMI_PYTHON)"
  [ -f "$FRONTEND_BUILD/index.html" ] || \
    die "frontend build missing: run (cd frontend && REACT_APP_API_BASE_URL= npm run build)"
  if grep -rqs --include='*.js' "localhost:8000" "$FRONTEND_BUILD/static/js"; then
    die "frontend build contains localhost:8000; rebuild with REACT_APP_API_BASE_URL="
  fi
  [ -d "$DATASET_DIR" ] || die "dataset directory not found: $DATASET_DIR"

  mkdir -p "$STATE_DIR"
  local existing
  existing="$(read_pid)"
  if [ -n "$existing" ] && kill -0 "$existing" 2>/dev/null; then
    die "demo already running (pid $existing). Run: deploy/local-demo/demo.sh stop"
  fi
  rm -f "$PID_FILE"

  if is_port_listening; then
    die "port $PORT on $HOST is already in use"
  fi

  echo "Starting ImgSimiSearch demo on $BASE_URL"
  echo "  python:          $PYTHON"
  echo "  dataset:         $DATASET_DIR"
  echo "  release store:   $RELEASE_STORE"
  echo "  frontend build:  $FRONTEND_BUILD"
  echo "  log:             $LOG_FILE"

  # exec keeps $! equal to the uvicorn pid; nohup survives terminal hangup.
  (
    cd "$BACKEND_DIR"
    exec env \
      IMGSIMI_DATASET_DIR="$DATASET_DIR" \
      IMGSIMI_RELEASE_STORE="$RELEASE_STORE" \
      IMGSIMI_FRONTEND_DIR="$FRONTEND_BUILD" \
      IMGSIMI_ENABLE_DOCS="${IMGSIMI_ENABLE_DOCS:-0}" \
      IMGSIMI_MAX_UPLOAD_BYTES="${IMGSIMI_MAX_UPLOAD_BYTES:-16777216}" \
      IMGSIMI_MAX_CONCURRENT_SEARCHES="${IMGSIMI_MAX_CONCURRENT_SEARCHES:-2}" \
      IMGSIMI_SEARCH_TIMEOUT_SECONDS="${IMGSIMI_SEARCH_TIMEOUT_SECONDS:-60}" \
      IMGSIMI_WARMUP_MODEL=1 \
      PYTHONUNBUFFERED=1 \
      nohup "$PYTHON" -m uvicorn main:app \
        --host "$HOST" --port "$PORT" \
        --workers 1 \
        --limit-concurrency 64 \
        --timeout-keep-alive 5 \
        --timeout-graceful-shutdown 10 \
        --no-server-header \
        >>"$LOG_FILE" 2>&1 < /dev/null
  ) &
  echo "$!" > "$PID_FILE"

  local waited=0
  until [ "$(http_code /health)" = "200" ]; do
    if ! kill -0 "$(read_pid)" 2>/dev/null; then
      rm -f "$PID_FILE"
      die "server exited during startup; see $LOG_FILE"
    fi
    if [ "$waited" -ge 240 ]; then
      die "server did not answer /health within 240s; see $LOG_FILE"
    fi
    sleep 1
    waited=$((waited + 1))
  done

  echo "Started (pid $(read_pid)). /health=200  /ready=$(http_code /ready)"
  echo "Open: $BASE_URL/"
  if [ "$(http_code /ready)" != "200" ]; then
    echo "WARNING: /ready is not 200. The runtime is not serving a validated release;"
    echo "         search results are not trustworthy. See $LOG_FILE."
  fi
}

cmd_status() {
  local pid
  pid="$(read_pid)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "process: running (pid $pid)"
  else
    echo "process: not running"
  fi
  echo "/health: $(http_code /health)"
  echo "/ready:  $(http_code /ready)"
}

cmd_stop() {
  local pid
  pid="$(read_pid)"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "demo is not running"
    return 0
  fi
  if ! pid_is_demo_server "$pid"; then
    die "pid $pid is not the ImgSimiSearch demo server on port $PORT; refusing to signal it"
  fi

  echo "Stopping demo server (pid $pid)..."
  kill -TERM "$pid"
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge 20 ]; then
      echo "pid $pid did not exit after 20s; sending SIGKILL" >&2
      kill -KILL "$pid" 2>/dev/null || true
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done
  rm -f "$PID_FILE"
  echo "stopped"
}

case "${1:-}" in
  start) cmd_start ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  *) echo "usage: $(basename "$0") {start|status|stop}" >&2; exit 2 ;;
esac
