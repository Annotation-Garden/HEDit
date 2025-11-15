# Docker Configuration

This directory contains Docker-related configuration files for HED-BOT.

## Files

### ollama-entrypoint.sh

Entrypoint script for the Ollama container that automatically:
1. Starts the Ollama server
2. Waits for it to be ready
3. Checks if the default model (`gpt-oss:20b`) exists
4. Pulls the model if not present
5. Keeps the server running

**Why automatic pulling?**
- Ensures the model is always available on first container start
- No manual intervention needed after `docker-compose up`
- Simplifies deployment process

**Model**: `gpt-oss:20b` (20 billion parameters, optimized for RTX 4090)

**Note**: The first container start will take longer (~10-20 minutes) while the model downloads. Subsequent starts are instant as the model is cached in the Docker volume.

## Usage

The entrypoint is automatically used by docker-compose.yml. No manual configuration needed.

To use a different model:
1. Update the `MODEL` variable in `ollama-entrypoint.sh`
2. Update `LLM_MODEL` in `.env` or `docker-compose.yml`
3. Restart containers: `docker-compose down && docker-compose up -d`
