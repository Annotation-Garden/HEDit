FROM python:3.13-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    curl \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22.x from NodeSource (required for HED JavaScript validator)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

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

# Copy requirements and install Python dependencies with API extras
COPY pyproject.toml README.md /app/
RUN pip install uv && \
    uv pip install --system --no-cache ".[api]"

# Copy the application code
COPY src /app/src

# Set environment variables for HED resources (internal paths)
ENV HED_SCHEMA_DIR=/app/hed-schemas/schemas_latest_json \
    HED_VALIDATOR_PATH=/app/hed-javascript \
    USE_JS_VALIDATOR=true

# Expose the application port
EXPOSE 38427

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:38427/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "38427", "--workers", "2"]
