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
#   0 * * * * /path/to/deploy/auto-update.sh >> /var/log/hed-bot-update.log 2>&1

##### Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${LOG_FILE:-/var/log/hed-bot-update.log}"
LOCK_FILE="/tmp/hed-bot-update.lock"

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
    IMAGE_NAME="hed-bot_dev:latest"
    CONTAINER_NAME="hed-bot_dev"
    REGISTRY_IMAGE="ghcr.io/neuromechanist/hed-bot:main"
else
    IMAGE_NAME="hed-bot:latest"
    CONTAINER_NAME="hed-bot"
    REGISTRY_IMAGE="ghcr.io/neuromechanist/hed-bot:latest"
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

    # Get current local image digest
    LOCAL_DIGEST=$(docker images --no-trunc --quiet "$IMAGE_NAME" 2>/dev/null)

    # Pull latest image metadata from registry
    log "Pulling latest image from registry: $REGISTRY_IMAGE"
    docker pull "$REGISTRY_IMAGE" > /dev/null 2>&1

    # Get new image digest
    NEW_DIGEST=$(docker images --no-trunc --quiet "$REGISTRY_IMAGE" 2>/dev/null)

    if [ -z "$NEW_DIGEST" ]; then
        error_exit "Failed to pull image from registry"
    fi

    # Tag the registry image with local name
    docker tag "$REGISTRY_IMAGE" "$IMAGE_NAME"

    if [ "$LOCAL_DIGEST" = "$NEW_DIGEST" ]; then
        log "No update available (digest: ${NEW_DIGEST:0:12})"
        return 1
    else
        log "New image available!"
        log "  Old digest: ${LOCAL_DIGEST:0:12}"
        log "  New digest: ${NEW_DIGEST:0:12}"
        return 0
    fi
}

# Deploy the new image
deploy_update() {
    log "Deploying update..."

    # Run deployment script
    if [ -f "${SCRIPT_DIR}/deploy.sh" ]; then
        bash "${SCRIPT_DIR}/deploy.sh" "$ENVIRONMENT" 127.0.0.1
        if [ $? -eq 0 ]; then
            log "✓ Deployment successful"
            return 0
        else
            error_exit "Deployment failed"
        fi
    else
        error_exit "deploy.sh not found at ${SCRIPT_DIR}/deploy.sh"
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
log "HED-BOT Auto-Update Check"
log "Environment: $ENVIRONMENT"
log "========================================="

# Acquire lock
acquire_lock
trap release_lock EXIT

# Check for updates
if check_for_updates || [ "$FORCE_UPDATE" = true ]; then
    if [ "$CHECK_ONLY" = true ]; then
        log "Check-only mode: Update available but not deploying"
        send_notification "HED-BOT update available for $ENVIRONMENT"
        exit 0
    fi

    # Deploy update
    deploy_update

    # Cleanup
    cleanup_old_images

    # Send success notification
    send_notification "HED-BOT $ENVIRONMENT successfully updated"

    log "========================================="
    log "Update completed successfully!"
    log "========================================="
else
    log "No updates needed"
fi

release_lock
