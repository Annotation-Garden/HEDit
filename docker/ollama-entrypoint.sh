#!/bin/bash
# Entrypoint script for Ollama container to automatically pull gpt-oss:20b model

set -e

# Start ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait for ollama to be ready
echo "Waiting for Ollama server to start..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama server is ready"
        break
    fi
    sleep 1
done

# Check if model exists, pull if not
MODEL="gpt-oss:20b"
echo "Checking if model $MODEL exists..."
if ! ollama list | grep -q "$MODEL"; then
    echo "Model $MODEL not found. Pulling..."
    ollama pull "$MODEL"
    echo "Model $MODEL pulled successfully"
else
    echo "Model $MODEL already exists"
fi

# Keep container running with ollama serve
wait $OLLAMA_PID
