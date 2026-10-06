#!/usr/bin/env bash
# Start the FastAPI backend and the Astro console. Ctrl+C stops both.
set -euo pipefail
set -m

root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"

if [[ ! -d web/node_modules ]]; then
  echo "Missing web/node_modules. From the repo root run: cd web && pnpm install" >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "No .env file. Copy .env.example and set FACE_ID_ADMIN_TOKEN, or People enrol/delete will be refused." >&2
fi

api_pid=""
web_pid=""

stop_group() {
  local pid=$1
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 0
  kill -- -"$pid" 2>/dev/null || true
  local _
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.2
  done
  kill -KILL -- -"$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_group "$api_pid"
  stop_group "$web_pid"
  exit "$status"
}
trap cleanup EXIT INT TERM

uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 &
api_pid=$!

pnpm --dir web dev &
web_pid=$!

echo "API      http://127.0.0.1:8000/docs"
echo "Console  http://127.0.0.1:4321"
echo "Ctrl+C stops both."

wait -n "$api_pid" "$web_pid"
