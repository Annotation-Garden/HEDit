#!/bin/bash
# sync-env.sh - Render the server .env from Infisical, the source of truth.
#
# Writes <repo>/.env (mode 0600) from the Infisical folder below, using the
# host's machine identity (Universal Auth). The deploy scripts call this right
# before they replace a container. If Infisical is unreachable or returns
# something unusable, the existing .env is kept and the failure is logged, so
# a deploy never starts from an empty or partial configuration.
#
# The .env is generated: edit values in Infisical (project annot-garden,
# prod /hedit/server), then redeploy (deploy/auto-update.sh --force).
# Prod and dev containers share this one file.
#
# Exit status: 0 when .env was rendered, 1 when the previous .env was kept.

INFISICAL_PROJECT_ID="4d8cc8dc-cdea-4826-86cd-b13ec79a8a82"   # annot-garden
INFISICAL_ENV="prod"
INFISICAL_PATH="/hedit/server"
# Keys without which the backend cannot serve requests
REQUIRED_KEYS="API_KEYS ANTHROPIC_API_KEY ANTHROPIC_BASE_URL ANTHROPIC_WORKSPACE_ID"

IDENTITY_FILE="${INFISICAL_IDENTITY_FILE:-${HOME}/.config/infisical/hedtools-identity.env}"
INFISICAL_BIN="${INFISICAL_BIN:-$(command -v infisical || echo "${HOME}/.local/bin/infisical")}"
TARGET="$(cd "$(dirname "$0")/.." && pwd)/.env"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] sync-env: $*"; }
keep_previous() {
    log "ERROR: $1; keeping the existing ${TARGET}"
    exit 1
}

[ -r "$IDENTITY_FILE" ] || keep_previous "identity file ${IDENTITY_FILE} is not readable"
[ -x "$INFISICAL_BIN" ] || keep_previous "infisical CLI not found (set INFISICAL_BIN)"

# The identity file provides INFISICAL_UNIVERSAL_AUTH_CLIENT_ID,
# INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET and INFISICAL_API_URL. The CLI reads
# them from the environment, so the secret never appears in a process listing.
set -a
# shellcheck disable=SC1090
. "$IDENTITY_FILE"
set +a

# A hung connection must not stall a deploy that holds its lock
INFISICAL_TOKEN="$(timeout 60 "$INFISICAL_BIN" login --method=universal-auth --domain="$INFISICAL_API_URL" --plain --silent 2>/dev/null)"
[ -n "$INFISICAL_TOKEN" ] || keep_previous "Infisical login failed"
export INFISICAL_TOKEN

umask 077
TMP="$(mktemp "${TARGET}.XXXXXX")" || keep_previous "cannot create a temporary file next to ${TARGET}"
trap 'rm -f "$TMP"' EXIT

# Docker's --env-file takes KEY=value literally (no quoting, no multi-line
# values), so build the file from the JSON export instead of the CLI's dotenv
# output, which quotes values.
timeout 60 "$INFISICAL_BIN" export --domain="$INFISICAL_API_URL" --projectId="$INFISICAL_PROJECT_ID" \
    --env="$INFISICAL_ENV" --path="$INFISICAL_PATH" --format=json --silent 2>/dev/null \
    | python3 -c '
import json, re, sys
required = sys.argv[1].split()
KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
try:
    secrets = json.load(sys.stdin)
except json.JSONDecodeError as error:
    sys.exit(f"export was not JSON ({error})")
values = {s["key"]: s["value"] for s in secrets}
problems = [f"{k} is missing or empty" for k in required if not values.get(k)]
problems += [f"key {k!r} is not a valid variable name" for k in values if not KEY.fullmatch(k)]
problems += [f"{k} spans several lines" for k, v in values.items() if "\n" in v or "\r" in v]
problems += [f"{k} is longer than docker --env-file accepts" for k, v in values.items() if len(k) + len(v) > 60000]
if problems:
    sys.exit("; ".join(problems))
for key in sorted(values):
    print(f"{key}={values[key]}")
' "$REQUIRED_KEYS" > "$TMP" 2>"${TMP}.err"
status=$?
reason="$(cat "${TMP}.err" 2>/dev/null)"
rm -f "${TMP}.err"
[ "$status" -eq 0 ] || keep_previous "Infisical export unusable: ${reason:-exit ${status}}"

chmod 600 "$TMP"
if [ -f "$TARGET" ] && cmp -s "$TMP" "$TARGET"; then
    chmod 600 "$TARGET"
    log "unchanged ($(wc -l < "$TMP") keys from ${INFISICAL_ENV}${INFISICAL_PATH})"
    exit 0
fi
mv "$TMP" "$TARGET" || keep_previous "cannot replace ${TARGET}"
log "rendered $(wc -l < "$TARGET") keys from ${INFISICAL_ENV}${INFISICAL_PATH}"
