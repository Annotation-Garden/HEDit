#!/bin/bash

# deploy.sh - Pull and deploy HEDit Docker container from GHCR
# Usage: ./deploy.sh [environment]
# Environment: 'prod' (default) or 'dev'

##### Constants
ENVIRONMENT="${1:-prod}"
DEPLOY_DIR=$(cd "$(dirname "$0")/.." && pwd)

# Set environment-specific variables
if [ "$ENVIRONMENT" = "dev" ]; then
    REGISTRY_IMAGE="ghcr.io/annotation-garden/hedit:dev"
    CONTAINER_NAME="hedit-dev"
    HOST_PORT=38428
else
    REGISTRY_IMAGE="ghcr.io/annotation-garden/hedit:latest"
    CONTAINER_NAME="hedit"
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

sync_env() {
    # Render .env from Infisical; on failure the existing .env is kept
    bash "${DEPLOY_DIR}/deploy/sync-env.sh" || echo "[WARN] Infisical sync failed; deploying with the existing .env"
    ENV_FILE="${DEPLOY_DIR}/.env"
    if [ ! -f "$ENV_FILE" ]; then
        error_exit "Missing env file: ${ENV_FILE}. Refusing to replace the container with one that has no configuration."
    fi
}

run_container() {
    echo "Starting container ${CONTAINER_NAME} on port ${HOST_PORT}..."

    ENV_ARGS="--env-file ${ENV_FILE}"

    # Create persistent feedback directory on host
    FEEDBACK_DIR="/var/lib/hedit/${CONTAINER_NAME}/feedback"
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
echo "HEDit Deployment (GHCR)"
echo "========================================="
echo "Environment: ${ENVIRONMENT}"
echo "Image: ${REGISTRY_IMAGE}"
echo "Container: ${CONTAINER_NAME}"
echo "Port: 127.0.0.1:${HOST_PORT}"
echo "========================================="

sync_env
pull_image
stop_existing_container
run_container
sleep 3
wait_for_health

echo ""
echo "Deployment complete!"
echo "Health: http://localhost:${HOST_PORT}/health"
echo "Logs:   docker logs -f ${CONTAINER_NAME}"
