#!/usr/bin/env bash
# Terraform — cross-platform launcher (macOS, WSL, Windows Git Bash).
#   ./run.sh start   — start the FastAPI dev server on http://127.0.0.1:8000/
#   ./run.sh stop    — stop the running server
#   ./run.sh         — same as start

set -u

cd "$(dirname "$0")" || exit 1

VENV="${VENV:-.venv}"
PY="${VENV}/bin/python"
UVICORD="${VENV}/bin/uvicorn"

BASE_DIR="$(pwd)"
PORT="${PORT:-8000}"
HOST="127.0.0.1"
ENV_FILE="${BASE_DIR}/.env"
DEFAULT_OFFLINE="1"

is_windows() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) return 0 ;;
    *) return 1 ;;
  esac
}

# Use the project .venv python when present, else fall back to the system python.
if [ -x "${PY}" ]; then
  PYTHON="${PY}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "No Python interpreter found (looked at ${VENV}/bin/python then python3)." >&2
  exit 1
fi

# Resolve the real python used by this environment so we can strip the
# legacy PYTHONPATH/PYTHONHOME that poisons the bundled site-packages.
REAL_PYTHON="$("${PYTHON}" -c "import sys; print(sys.executable)")"

# --- start ---
start() {
  if [ -f "${BASE_DIR}/.run.pid" ]; then
    OLD="$(cat "${BASE_DIR}/.run.pid")"
    if kill -0 "${OLD}" 2>/dev/null; then
      echo "Server already running (pid ${OLD})."
      return 0
    fi
    rm -f "${BASE_DIR}/.run.pid"
  fi

  # Windows shells sometimes carry a stale PYTHONPATH/PYTHONHOME that points at
  # an incompatible (Python 3.14) site-packages. Strip it so uvicorn uses the
  # project .venv instead of the global interpreter.
  unset PYTHONPATH
  unset PYTHONHOME

  nohup "${PYTHON}" -m uvicorn src.api.main:app \
    --host "${HOST}" --port "${PORT}" > "${BASE_DIR}/terra-nova.log" 2>&1 &
  echo $! > "${BASE_DIR}/.run.pid"
  echo "Started server on http://${HOST}:${PORT}/ (pid $(cat "${BASE_DIR}/.run.pid"))."
}

# --- stop ---
stop() {
  if [ ! -f "${BASE_DIR}/.run.pid" ]; then
    echo "No running server (.run.pid not found)."
    return 0
  fi
  PID="$(cat "${BASE_DIR}/.run.pid")"
  if kill -0 "${PID}" 2>/dev/null; then
    kill "${PID}" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "${PID}" 2>/dev/null || break
      sleep 1
    done
    kill -9 "${PID}" 2>/dev/null
  fi
  rm -f "${BASE_DIR}/.run.pid"
  echo "Stopped server (was pid ${PID})."
}

case "${1:-start}" in
  start) start ;;
  stop)  stop ;;
  *) echo "Usage: ./run.sh [start|stop]" >&2; exit 1 ;;
esac
