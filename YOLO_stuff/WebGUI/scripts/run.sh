#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtual environment at $VENV_DIR" >&2
  echo "Create it with: python3 -m venv venv" >&2
  exit 1
fi

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "Missing requirements file at $REQUIREMENTS_FILE" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import flask, mavsdk, serial" >/dev/null 2>&1; then
  echo "Installing Python dependencies from $REQUIREMENTS_FILE"
  "$PIP_BIN" install -r "$REQUIREMENTS_FILE"
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Note: app.py binds to port 67, which is privileged on Unix." >&2
  echo "Run with sudo or change WEB_PORT in app.py if startup fails." >&2
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/app.py" "$@"
