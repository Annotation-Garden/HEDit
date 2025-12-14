#!/bin/bash

# deploy.sh - Pull and deploy HED-BOT Docker container from GHCR
# Usage: ./deploy.sh [environment]
# Environment: 'prod' (default) or 'dev'

##### Constants
ENVIRONMENT="${1:-prod}"
DEPLOY_DIR=$(cd "$(dirname "$0")/.." && pwd)

# Set environment-specific variables
if [ "$ENVIRONMENT" = "dev" ]; then
    REGISTRY_IMAGE="ghcr.io/neuromechanist/hed-bot:dev"
    CONTAINER_NAME="hed-bot-dev"
    HOST_PORT=38428
else
    REGISTRY_IMAGE="ghcr.io/neuromechanist/hed-bot:latest"
    CONTAINER_NAME="hed-bot"
    HOST_PORT=38427
fi

CONTAINER_PORT=38427

##### Functions

error_exit() {
    echo "[ERROR] $1"
    exit 1
}

pull_image() {
    echo "Pulling image from registry: ${REGISTRY_IMAGE}..."
    docker pull "${REGISTRY_IMAGE}" || error_exit "Failed to pull image"
}

stop_existing_container() {
    echo "Stopping existing container ${CONTAINER_NAME}..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
}

run_container() {
    echo "Starting container ${CONTAINER_NAME} on port ${HOST_PORT}..."

    ENV_FILE="${DEPLOY_DIR}/.env"
    ENV_ARGS=""
    if [ -f "$ENV_FILE" ]; then
        ENV_ARGS="--env-file ${ENV_FILE}"
    fi

    # Create persistent feedback directory on host
    FEEDBACK_DIR="/var/lib/hed-bot/${CONTAINER_NAME}/feedback"
    mkdir -p "${FEEDBACK_DIR}/unprocessed" "${FEEDBACK_DIR}/processed" 2>/dev/null || \
        echo "Warning: Could not create ${FEEDBACK_DIR}, feedback may not persist"

    docker run -d \
        --name "${CONTAINER_NAME}" \
        --restart unless-stopped \
        -p "127.0.0.1:${HOST_PORT}:${CONTAINER_PORT}" \
        -v "${FEEDBACK_DIR}:/app/feedback" \
        ${ENV_ARGS} \
        "${REGISTRY_IMAGE}" || error_exit "Failed to start container"
}

wait_for_health() {
    echo "Waiting for container to be healthy..."
    for i in {1..30}; do
        if curl -sf "http://localhost:${HOST_PORT}/health" > /dev/null 2>&1; then
            echo "Container is healthy!"
            return 0
        fi
        sleep 2
    done
    echo "Warning: Container did not become healthy within timeout"
    return 1
}

##### Main
echo "========================================="
echo "HED-BOT Deployment (GHCR)"
echo "========================================="
echo "Environment: ${ENVIRONMENT}"
echo "Image: ${REGISTRY_IMAGE}"
echo "Container: ${CONTAINER_NAME}"
echo "Port: 127.0.0.1:${HOST_PORT}"
echo "========================================="

pull_image
stop_existing_container
run_container
sleep 3
wait_for_health

echo ""
echo "Deployment complete!"
echo "Health: http://localhost:${HOST_PORT}/health"
echo "Logs:   docker logs -f ${CONTAINER_NAME}"
