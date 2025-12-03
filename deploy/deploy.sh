#!/bin/bash

# deploy.sh - Script to build and deploy HED-BOT Docker container
# Usage: ./deploy.sh [environment] [bind_address]
# Environment can be 'prod' or 'dev' (defaults to 'prod')
# bind_address can be an IP like 0.0.0.0 (default) or 127.0.0.1 to restrict to localhost

##### Constants
ENVIRONMENT="${1:-prod}"
BIND_ADDRESS="${2:-127.0.0.1}"
DEPLOY_DIR=$(cd "$(dirname "$0")/.." && pwd)

# Set environment-specific variables
if [ "$ENVIRONMENT" = "dev" ]; then
    IMAGE_NAME="hed-bot_dev:latest"
    CONTAINER_NAME="hed-bot_dev"
    HOST_PORT=38428
    URL_PREFIX="/hed-bot-api-dev"
else
    IMAGE_NAME="hed-bot:latest"
    CONTAINER_NAME="hed-bot"
    HOST_PORT=38427
    URL_PREFIX="/hed-bot-api"
fi

CONTAINER_PORT=38427

##### Functions

# Print error message and exit
error_exit() {
    echo "[ERROR] $1"
    exit 1
}

# Build the Docker image
build_docker_image() {
    echo "Building Docker image ${IMAGE_NAME} for ${ENVIRONMENT} environment..."
    cd "${DEPLOY_DIR}"
    docker build -f deploy/Dockerfile -t "${IMAGE_NAME}" . || error_exit "Failed to build Docker image"
}

# Stop and remove existing container
stop_existing_container() {
    echo "Stopping and removing existing container ${CONTAINER_NAME}..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null || echo "Container ${CONTAINER_NAME} was not running"
    docker rm "${CONTAINER_NAME}" 2>/dev/null || echo "Container ${CONTAINER_NAME} did not exist"
}

# Run the Docker container
run_docker_container() {
    echo "Running Docker container ${CONTAINER_NAME} on ${BIND_ADDRESS}:${HOST_PORT}..."

    # Check if .env file exists
    ENV_FILE="${DEPLOY_DIR}/.env"
    if [ ! -f "$ENV_FILE" ]; then
        echo "Warning: .env file not found at ${ENV_FILE}"
        echo "Using environment variables from shell"
        ENV_ARGS=""
    else
        ENV_ARGS="--env-file ${ENV_FILE}"
    fi

    docker run -d \
        --name "${CONTAINER_NAME}" \
        --restart unless-stopped \
        -p "${BIND_ADDRESS}:${HOST_PORT}:${CONTAINER_PORT}" \
        ${ENV_ARGS} \
        -e HED_URL_PREFIX="${URL_PREFIX}" \
        "${IMAGE_NAME}" || error_exit "Failed to run Docker container"
}

# Show container logs
show_logs() {
    echo "Showing container logs (last 20 lines)..."
    docker logs --tail 20 "${CONTAINER_NAME}"
}

# Wait for health check
wait_for_health() {
    echo "Waiting for container to be healthy..."
    for i in {1..30}; do
        if docker inspect --format='{{.State.Health.Status}}' "${CONTAINER_NAME}" 2>/dev/null | grep -q "healthy"; then
            echo "Container is healthy!"
            return 0
        fi
        echo "Waiting... ($i/30)"
        sleep 2
    done
    echo "Warning: Container did not become healthy within timeout"
    return 1
}

##### Main execution
echo "========================================="
echo "HED-BOT Deployment"
echo "========================================="
echo "Environment: ${ENVIRONMENT}"
echo "Image: ${IMAGE_NAME}"
echo "Container: ${CONTAINER_NAME}"
echo "Port: ${BIND_ADDRESS}:${HOST_PORT} -> ${CONTAINER_PORT}"
echo "========================================="

build_docker_image
stop_existing_container
run_docker_container

# Wait a moment for container to start
sleep 3

# Check if container is running
if docker ps | grep -q "${CONTAINER_NAME}"; then
    echo "✓ Container is running"
    show_logs

    # Wait for health check if available
    if docker inspect --format='{{.State.Health}}' "${CONTAINER_NAME}" 2>/dev/null | grep -q "Starting"; then
        wait_for_health
    fi

    echo "========================================="
    echo "Deployment completed successfully!"
    echo "Application is running at: http://localhost:${HOST_PORT}"
    echo "Health check: http://localhost:${HOST_PORT}/health"
    echo "========================================="
    echo ""
    echo "Useful commands:"
    echo "  View logs:    docker logs -f ${CONTAINER_NAME}"
    echo "  Stop:         docker stop ${CONTAINER_NAME}"
    echo "  Restart:      docker restart ${CONTAINER_NAME}"
    echo "  Remove:       docker rm -f ${CONTAINER_NAME}"
else
    echo "✗ Container failed to start"
    echo "Checking logs..."
    docker logs "${CONTAINER_NAME}"
    error_exit "Container did not start successfully"
fi
