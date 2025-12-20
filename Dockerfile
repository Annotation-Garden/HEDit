# Multi-stage Dockerfile for HEDit with GPU support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies and NodeSource for Node.js 22.x
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22.x from NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Set working directory
WORKDIR /app

# Clone HED repositories (self-contained)
# Using schemas_latest_json from official hed-schemas repo (JSON inheritance fix now merged)
RUN git clone --depth 1 https://github.com/hed-standard/hed-schemas.git /app/hed-schemas && \
    git clone --depth 1 https://github.com/hed-standard/hed-javascript.git /app/hed-javascript

# Build HED JavaScript validator
WORKDIR /app/hed-javascript
RUN npm install && npm run build

# Return to app directory
WORKDIR /app

# Build argument for commit hash (set during CI build)
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install Python dependencies with API extras (includes uvicorn, FastAPI, etc.)
RUN pip install uv && \
    uv pip install --system --no-cache ".[api]"

# Set environment variables for HED resources (internal paths)
ENV HED_SCHEMA_DIR=/app/hed-schemas/schemas_latest_json \
    HED_VALIDATOR_PATH=/app/hed-javascript \
    USE_JS_VALIDATOR=true

# Expose port
EXPOSE 38427

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:38427/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "38427"]
