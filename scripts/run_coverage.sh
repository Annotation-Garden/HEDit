#!/bin/bash
# Run all tests with combined coverage
# Usage: ./scripts/run_coverage.sh [--standalone] [--all]
#
# Options:
#   (no args)     Run unit tests only (fast, no API key needed)
#   --standalone  Run unit + standalone tests (needs OPENROUTER_API_KEY_FOR_TESTING)
#   --all         Run all tests including integration (needs API key)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Load .env file if it exists
if [ -f .env ]; then
    echo -e "${GREEN}Loading environment from .env${NC}"
    set -a
    source .env
    set +a
fi

# Clean previous coverage data
echo -e "${YELLOW}Cleaning previous coverage data...${NC}"
rm -f .coverage .coverage.*
rm -rf htmlcov

# Parse arguments
RUN_STANDALONE=false
RUN_ALL=false

for arg in "$@"; do
    case $arg in
        --standalone)
            RUN_STANDALONE=true
            ;;
        --all)
            RUN_ALL=true
            RUN_STANDALONE=true
            ;;
    esac
done

# Step 1: Run unit tests (excluding integration and standalone)
echo -e "\n${GREEN}=== Running unit tests ===${NC}"
pytest tests/ -v -m "not integration and not standalone" \
    --ignore=tests/test_local_executor.py \
    --ignore=tests/test_api_security.py \
    --ignore=tests/test_workflow_api.py \
    --cov=src --cov-report= || true

# Step 2: Run standalone tests if requested
if [ "$RUN_STANDALONE" = true ]; then
    if [ -n "$OPENROUTER_API_KEY_FOR_TESTING" ] || [ -n "$OPENROUTER_API_KEY" ]; then
        # Use OPENROUTER_API_KEY as fallback
        if [ -z "$OPENROUTER_API_KEY_FOR_TESTING" ]; then
            export OPENROUTER_API_KEY_FOR_TESTING="$OPENROUTER_API_KEY"
        fi

        echo -e "\n${GREEN}=== Running standalone tests (real LLM calls) ===${NC}"
        pytest tests/ -v -m standalone --timeout=180 \
            --cov=src --cov-append --cov-report= || true
    else
        echo -e "\n${YELLOW}Skipping standalone tests: OPENROUTER_API_KEY_FOR_TESTING not set${NC}"
    fi
fi

# Step 3: Run integration tests if requested
if [ "$RUN_ALL" = true ]; then
    if [ -n "$OPENROUTER_API_KEY_FOR_TESTING" ]; then
        echo -e "\n${GREEN}=== Running integration tests (real LLM calls) ===${NC}"
        pytest tests/ -v -m integration --timeout=180 \
            --cov=src --cov-append --cov-report= || true
    else
        echo -e "\n${YELLOW}Skipping integration tests: OPENROUTER_API_KEY_FOR_TESTING not set${NC}"
    fi
fi

# Step 4: Combine coverage and generate reports
echo -e "\n${GREEN}=== Generating coverage reports ===${NC}"
coverage combine 2>/dev/null || true
coverage report --show-missing
coverage html

echo -e "\n${GREEN}=== Coverage report generated ===${NC}"
echo -e "View HTML report: ${YELLOW}open htmlcov/index.html${NC}"
