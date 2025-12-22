#!/bin/bash

# auto-update.sh - Automatic Docker image update script
# This script checks for new Docker images and automatically updates the running container
#
# Usage:
#   ./auto-update.sh [options]
#
# Options:
#   --check-only    Only check for updates, don't deploy
#   --force         Force update even if no new image available
#   --env ENV       Environment (prod|dev), default: prod
#
# Setup as cron job (check every hour):
#   0 * * * * /path/to/deploy/auto-update.sh >> /var/log/hedit/auto-update.log 2>&1

##### Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${LOG_FILE:-/var/log/hedit/auto-update.log}"
LOCK_FILE="/tmp/hedit-update.lock"

# Default values
CHECK_ONLY=false
FORCE_UPDATE=false
ENVIRONMENT="prod"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --force)
            FORCE_UPDATE=true
            shift
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set environment-specific variables
if [ "$ENVIRONMENT" = "dev" ]; then
    IMAGE_NAME="hedit-dev:latest"
    CONTAINER_NAME="hedit-dev"
    REGISTRY_IMAGE="ghcr.io/annotation-garden/hedit:dev"
else
    IMAGE_NAME="hedit:latest"
    CONTAINER_NAME="hedit"
    REGISTRY_IMAGE="ghcr.io/annotation-garden/hedit:latest"
fi

##### Functions

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Acquire lock to prevent concurrent updates
acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        LOCK_PID=$(cat "$LOCK_FILE")
        if ps -p "$LOCK_PID" > /dev/null 2>&1; then
            log "Update already in progress (PID: $LOCK_PID)"
            exit 0
        else
            log "Stale lock file found, removing"
            rm -f "$LOCK_FILE"
        fi
    fi
    echo $$ > "$LOCK_FILE"
}

release_lock() {
    rm -f "$LOCK_FILE"
}

# Check if new image is available
check_for_updates() {
    log "Checking for updates..."

    # Get the image digest of the RUNNING container (not just the local tag)
    RUNNING_IMAGE=$(docker inspect "$CONTAINER_NAME" --format='{{.Image}}' 2>/dev/null || echo "")
    if [ -z "$RUNNING_IMAGE" ]; then
        log "Container $CONTAINER_NAME not running, will deploy fresh"
        RUNNING_DIGEST=""
    else
        RUNNING_DIGEST="$RUNNING_IMAGE"
        log "Running container image: ${RUNNING_DIGEST:0:19}..."
    fi

    # Pull latest image from registry
    log "Pulling latest image from registry: $REGISTRY_IMAGE"
    docker pull "$REGISTRY_IMAGE" > /dev/null 2>&1

    # Get new image digest (full sha256)
    NEW_DIGEST=$(docker inspect "$REGISTRY_IMAGE" --format='{{.Id}}' 2>/dev/null)

    if [ -z "$NEW_DIGEST" ]; then
        error_exit "Failed to pull image from registry"
    fi

    log "Latest registry image: ${NEW_DIGEST:0:19}..."

    # Tag the registry image with local name
    docker tag "$REGISTRY_IMAGE" "$IMAGE_NAME"

    if [ "$RUNNING_DIGEST" = "$NEW_DIGEST" ]; then
        log "No update available (container already running latest)"
        return 1
    else
        log "New image available!"
        log "  Running: ${RUNNING_DIGEST:0:19}..."
        log "  Latest:  ${NEW_DIGEST:0:19}..."
        return 0
    fi
}

# Deploy the new image
deploy_update() {
    log "Deploying update..."

    # Find .env file
    ENV_FILE="${SCRIPT_DIR}/../.env"
    if [ ! -f "$ENV_FILE" ]; then
        ENV_FILE="${SCRIPT_DIR}/.env"
    fi

    ENV_ARGS=""
    if [ -f "$ENV_FILE" ]; then
        ENV_ARGS="--env-file ${ENV_FILE}"
        log "Using env file: ${ENV_FILE}"
    else
        log "Warning: No .env file found"
    fi

    # Set port based on environment
    if [ "$ENVIRONMENT" = "dev" ]; then
        HOST_PORT=38428
    else
        HOST_PORT=38427
    fi

    # Stop and remove existing container
    log "Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true

    # Create persistent feedback directory
    FEEDBACK_DIR="/var/lib/hedit/${CONTAINER_NAME}/feedback"
    mkdir -p "${FEEDBACK_DIR}/unprocessed" "${FEEDBACK_DIR}/processed" 2>/dev/null || true
    log "Feedback directory: ${FEEDBACK_DIR}"

    # Create persistent telemetry directory
    TELEMETRY_DIR="/var/lib/hedit/${CONTAINER_NAME}/telemetry"
    mkdir -p "${TELEMETRY_DIR}" 2>/dev/null || true
    log "Telemetry directory: ${TELEMETRY_DIR}"

    # Run the new container using the pulled image
    log "Starting new container on port ${HOST_PORT}..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        -p "127.0.0.1:${HOST_PORT}:38427" \
        ${ENV_ARGS} \
        -v /var/log/hedit:/var/log/hedit \
        -v "${FEEDBACK_DIR}:/app/feedback" \
        -v "${TELEMETRY_DIR}:/app/telemetry" \
        "$REGISTRY_IMAGE"

    if [ $? -eq 0 ]; then
        log "✓ Container started successfully"

        # Wait for health check
        log "Waiting for container to be healthy..."
        for i in {1..30}; do
            if docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "healthy"; then
                log "✓ Container is healthy"
                return 0
            fi
            sleep 2
        done
        log "Warning: Container did not become healthy within timeout, but it's running"
        return 0
    else
        error_exit "Failed to start container"
    fi
}

# Cleanup old Docker images
cleanup_old_images() {
    log "Cleaning up old images..."
    docker image prune -f --filter "dangling=true" > /dev/null 2>&1
    log "✓ Cleanup complete"
}

# Send notification (optional - can integrate with email, Slack, etc.)
send_notification() {
    MESSAGE="$1"
    # Placeholder for notification system
    # Example: curl -X POST -d "message=$MESSAGE" https://your-notification-endpoint
    log "NOTIFICATION: $MESSAGE"
}

##### Main execution
log "========================================="
log "HEDit Auto-Update Check"
log "Environment: $ENVIRONMENT"
log "========================================="

# Acquire lock
acquire_lock
trap release_lock EXIT

# Check for updates
if check_for_updates || [ "$FORCE_UPDATE" = true ]; then
    if [ "$CHECK_ONLY" = true ]; then
        log "Check-only mode: Update available but not deploying"
        send_notification "HEDit update available for $ENVIRONMENT"
        exit 0
    fi

    # Deploy update
    deploy_update

    # Cleanup
    cleanup_old_images

    # Send success notification
    send_notification "HEDit $ENVIRONMENT successfully updated"

    log "========================================="
    log "Update completed successfully!"
    log "========================================="
else
    log "No updates needed"
fi

release_lock
