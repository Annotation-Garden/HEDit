#!/bin/bash
# Deployment script for HED-BOT
# This script rebuilds and redeploys the Docker containers with latest code

set -e  # Exit on error

echo "========================================="
echo "HED-BOT Deployment Script"
echo "========================================="
echo ""

# Parse arguments
REBUILD_FORCE=false
NO_CACHE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            REBUILD_FORCE=true
            NO_CACHE=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force, -f     Force rebuild with no cache (slower but ensures clean build)"
            echo "  --no-cache      Rebuild without cache"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0              # Normal deployment (uses cache)"
            echo "  $0 --force      # Clean rebuild (recommended after code changes)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Step 1: Stop existing containers
echo "Step 1: Stopping existing containers..."
docker compose down
echo "✓ Containers stopped"
echo ""

# Step 2: Rebuild hed-bot image
echo "Step 2: Rebuilding hed-bot image..."
if [ "$NO_CACHE" = true ]; then
    echo "  (Building without cache - this may take a few minutes)"
    docker compose build --no-cache hed-bot
else
    echo "  (Building with cache)"
    docker compose build hed-bot
fi
echo "✓ Image rebuilt"
echo ""

# Step 3: Start services
echo "Step 3: Starting services..."
docker compose up -d
echo "✓ Services started"
echo ""

# Step 4: Wait for health checks
echo "Step 4: Waiting for services to be healthy..."
sleep 5

# Check hed-bot health
echo -n "  Checking hed-bot API... "
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:38427/health &> /dev/null; then
        echo "✓ Healthy"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "✗ Failed to become healthy"
        echo ""
        echo "Check logs with: docker-compose logs hed-bot"
        exit 1
    fi
    sleep 2
done

# Fetch and display version
echo ""
echo "========================================="
echo "Deployment Summary"
echo "========================================="
echo -n "API Version: "
curl -s http://localhost:38427/version | python3 -c "import sys, json; print(json.load(sys.stdin)['version'])" 2>/dev/null || echo "Unable to fetch"
echo ""
echo "API URL: http://localhost:38427"
echo "Health: http://localhost:38427/health"
echo ""
echo "View logs: docker-compose logs -f hed-bot"
echo "Stop services: docker-compose down"
echo "========================================="
echo ""
echo "✓ Deployment complete!"
