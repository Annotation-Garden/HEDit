#!/usr/bin/env bash
# Smoke test for the frontend Worker: runs it under `wrangler dev` (the real
# local runtime and asset handler, no fake bindings) and checks the mount.
# Usage: bash workers/web/smoke-test.sh
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8799}"
BASE="http://127.0.0.1:${PORT}"
LOG="$(mktemp)"

bunx "wrangler@${WRANGLER_VERSION:-4.149.0}" dev --port "$PORT" --ip 127.0.0.1 >"$LOG" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; rm -f "$LOG"' EXIT

for _ in $(seq 1 60); do
  curl -s -o /dev/null "$BASE/" && break
  sleep 1
done
curl -s -o /dev/null "$BASE/" || { echo "wrangler dev did not start"; cat "$LOG"; exit 1; }

failures=0
check() { # description expected actual
  if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1 (expected '$2', got '$3')"; failures=$((failures + 1)); fi
}
status() { curl -s -o /dev/null -w '%{http_code}' "$BASE$1"; }
location() { curl -s -o /dev/null -w '%{redirect_url}' "$BASE$1"; }

check "/hedit/ serves the page" 200 "$(status /hedit/)"
page="$(curl -s "$BASE/hedit/")"
check "/hedit/ is the HEDit page" yes "$(grep -q '<title>HEDit' <<<"$page" && echo yes || echo no)"
check "/hedit/config.js is served" 200 "$(status /hedit/config.js)"
check "/hedit/assets/HED.svg is served" 200 "$(status /hedit/assets/HED.svg)"
check "bare /hedit redirects" 308 "$(status '/hedit?x=1')"
check "bare /hedit keeps the query" "$BASE/hedit/?x=1" "$(location '/hedit?x=1')"
check "/hedit/index.html redirect stays under the mount" "$BASE/hedit/?x=1" "$(location '/hedit/index.html?x=1')"
check "missing asset is 404" 404 "$(status /hedit/nope.js)"
check "path outside the mount is 404" 404 "$(status /hedit-dev/)"
check "root is outside the mount" 404 "$(status /)"

if [ "$failures" -gt 0 ]; then
  echo "$failures check(s) failed"
  exit 1
fi
echo "All frontend Worker checks passed"
