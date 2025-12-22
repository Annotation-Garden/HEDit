#!/bin/bash
# auto-update-dev.sh - Minimal auto-update for dev channel
# Cron: */15 * * * * /path/to/deploy/auto-update-dev.sh >> /var/log/hedit/auto-update-dev.log 2>&1

REGISTRY_IMAGE="ghcr.io/annotation-garden/hedit:dev"
CONTAINER_NAME="hedit-dev"
HOST_PORT=38428
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCK_FILE="/tmp/hedit-dev-update.lock"

# Prevent concurrent runs
[ -f "$LOCK_FILE" ] && exit 0
echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

# Get current container image ID
RUNNING_ID=$(docker inspect "$CONTAINER_NAME" --format='{{.Image}}' 2>/dev/null)

# Pull latest
docker pull "$REGISTRY_IMAGE" > /dev/null 2>&1 || { rm -f "$LOCK_FILE"; exit 1; }

# Get new image ID
NEW_ID=$(docker inspect "$REGISTRY_IMAGE" --format='{{.Id}}' 2>/dev/null)

# Compare and update if different
if [ "$RUNNING_ID" != "$NEW_ID" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M')] Updating dev container..."

    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true

    ENV_FILE="${SCRIPT_DIR}/../.env"
    ENV_ARGS=""
    [ -f "$ENV_FILE" ] && ENV_ARGS="--env-file $ENV_FILE"

    # Create persistent feedback directory
    FEEDBACK_DIR="/var/lib/hedit/${CONTAINER_NAME}/feedback"
    mkdir -p "${FEEDBACK_DIR}/unprocessed" "${FEEDBACK_DIR}/processed" 2>/dev/null || true

    # Create persistent telemetry directory
    TELEMETRY_DIR="/var/lib/hedit/${CONTAINER_NAME}/telemetry"
    mkdir -p "${TELEMETRY_DIR}" 2>/dev/null || true

    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        -p "127.0.0.1:${HOST_PORT}:38427" \
        -v "${FEEDBACK_DIR}:/app/feedback" \
        -v "${TELEMETRY_DIR}:/app/telemetry" \
        $ENV_ARGS \
        "$REGISTRY_IMAGE" > /dev/null

    echo "[$(date '+%Y-%m-%d %H:%M')] Dev updated: ${NEW_ID:7:12}"
fi
