#!/bin/bash
# Local development server for HEDit
# Usage: ./scripts/dev.sh [--no-auth]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}HEDit Local Development Server${NC}"
echo -e "${GREEN}========================================${NC}"

# Parse arguments
NO_AUTH=false
for arg in "$@"; do
    case $arg in
        --no-auth)
            NO_AUTH=true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-auth    Disable API key authentication"
            echo "  --help       Show this help message"
            exit 0
            ;;
    esac
done

# Check for .env file
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${YELLOW}Warning: No .env file found${NC}"
    echo "Creating from .env.example..."
    if [ -f "$PROJECT_DIR/.env.example" ]; then
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        echo -e "${YELLOW}Please edit .env with your settings${NC}"
    else
        echo -e "${RED}No .env.example found. Create .env manually.${NC}"
    fi
fi

# Load environment
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Override settings based on flags
if [ "$NO_AUTH" = true ]; then
    export REQUIRE_API_AUTH=false
    echo -e "${YELLOW}API authentication disabled${NC}"
fi

# Check for required environment variables
if [ -z "$OPENROUTER_API_KEY" ] && [ "$LLM_PROVIDER" = "openrouter" ]; then
    echo -e "${RED}Error: OPENROUTER_API_KEY not set${NC}"
    echo "Set it in .env or export it:"
    echo "  export OPENROUTER_API_KEY=your-key"
    exit 1
fi

# Set local paths for HED resources
export HED_SCHEMA_DIR="${HED_SCHEMA_DIR:-$HOME/Documents/git/HED/hed-schemas/schemas_latest_json}"
export HED_VALIDATOR_PATH="${HED_VALIDATOR_PATH:-$HOME/Documents/git/HED/hed-javascript}"

# Verify paths exist
if [ ! -d "$HED_SCHEMA_DIR" ]; then
    echo -e "${RED}Error: HED schema directory not found: $HED_SCHEMA_DIR${NC}"
    exit 1
fi

if [ ! -d "$HED_VALIDATOR_PATH" ]; then
    echo -e "${RED}Error: HED validator not found: $HED_VALIDATOR_PATH${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  LLM Provider: ${LLM_PROVIDER:-ollama}"
echo "  Schema Dir: $HED_SCHEMA_DIR"
echo "  Validator: $HED_VALIDATOR_PATH"
echo "  Auth Required: ${REQUIRE_API_AUTH:-true}"
echo ""
echo -e "${GREEN}Starting server on http://localhost:38427${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Change to project directory and run
cd "$PROJECT_DIR"
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 38427
