# Docker Architecture (Local Development)

> **Note**: This document describes the Docker architecture for local development. Local Ollama-based inference has been removed; LLM calls go to the Claude Platform on AWS (see **[claude-platform-aws.md](claude-platform-aws.md)**). For production deployment architecture, see **[deploy/DEPLOYMENT_ARCHITECTURE.md](../../deploy/DEPLOYMENT_ARCHITECTURE.md)**.

## Overview

HEDit uses a **fully self-contained Docker architecture** for local development that requires no external dependencies on the host system.

## Container Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                        │
├─────────────────────────────────────────────────────────┤
│  HEDit API Container                                   │
│  ├─ Python 3.11                                          │
│  ├─ FastAPI Backend                                      │
│  ├─ LangGraph Agents                                     │
│  ├─ HED Schemas (embedded)                               │
│  └─ HED Validator (embedded)                             │
└─────────────────────────────────────────────────────────┘
         │
         └── HTTPS ──> Claude Platform on AWS (Anthropic Messages API)
```

## Self-Contained Design

### What's Included in the Image

1. **Python Environment**
   - Python 3.11
   - All Python dependencies (LangGraph, LangChain, FastAPI, etc.)
   - Installed as a package (not editable mode)

2. **HED Resources** (Cloned from GitHub during build)
   - `hed-schemas` repository → `/app/hed-schemas`
   - `hed-javascript` repository → `/app/hed-javascript`
   - JavaScript validator built during image build

3. **System Dependencies**
   - Node.js 18+ (for validator)
   - npm (for building validator)
   - curl (for health checks)
   - git (for cloning HED repos)

### What's NOT Needed

- ❌ Local HED schemas directory
- ❌ Local HED JavaScript validator
- ❌ Host-side Python installation
- ❌ Host-side Node.js installation
- ❌ Volume mounts for source code

## Build Process

### Dockerfile Steps

```dockerfile
# 1. Base image with CUDA support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 2. Install system dependencies
RUN apt-get install python3.11 nodejs npm git curl

# 3. Clone HED resources (self-contained)
RUN git clone hed-schemas /app/hed-schemas
RUN git clone hed-javascript /app/hed-javascript

# 4. Build JavaScript validator
RUN cd /app/hed-javascript && npm install && npm run build

# 5. Copy application code
COPY src/ ./src/

# 6. Install application (NOT editable mode)
RUN pip install .

# 7. Set environment variables (internal paths)
ENV HED_SCHEMA_DIR=/app/hed-schemas/schemas_latest_json
ENV HED_VALIDATOR_PATH=/app/hed-javascript
```

### Why Not Editable Install?

**Before** (`pip install -e .`):
- Source code mounted from host
- Changes on host reflected immediately
- Can cause dependency resolution issues in Docker
- Not truly containerized

**After** (`pip install .`):
- Code copied into image during build
- Fully self-contained
- No volume mounts needed
- Production-ready
- Cleaner dependency resolution

## Environment Variables

### Set in Dockerfile (Defaults)

```bash
HED_SCHEMA_DIR=/app/hed-schemas/schemas_latest_json
HED_VALIDATOR_PATH=/app/hed-javascript
USE_JS_VALIDATOR=true
```

### Set in docker-compose.yml

```yaml
environment:
  - LLM_PROVIDER=anthropic
  - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
  - ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}
  - ANTHROPIC_WORKSPACE_ID=${ANTHROPIC_WORKSPACE_ID}
  - ANNOTATION_MODEL=claude-haiku-4-5
  - HED_SCHEMA_VERSION=8.3.0
```

### Override at Runtime (Optional)

```bash
docker run -e HED_SCHEMA_VERSION=8.4.0 hedit-api
```

## Build Optimization

### .dockerignore

Excludes unnecessary files from build context:
- `.git/` - Git history not needed
- `docs/` - Documentation not needed in production
- `tests/` - Tests run during development, not in container
- `frontend/` - Served separately
- `.env` - Secrets not in image

**Result**: Faster builds, smaller context

### Image Layers

1. Base system (CUDA) - **Rarely changes**
2. System packages - **Rarely changes**
3. HED resources - **Changes when repos update**
4. Application code - **Changes frequently**

Docker caches layers, so rebuilds are fast when only app code changes.

## Deployment Workflow

### First Deployment

```bash
# 1. Clone repository
git clone https://github.com/Annotation-Garden/HEDit.git && cd HEDit

# 2. Build images (includes HED resources)
docker-compose build

# 3. Start containers
docker-compose up -d

# 4. Monitor startup
docker-compose logs -f

# 5. Verify
curl http://localhost:38427/health
```

### Updating Application Code

```bash
# 1. Pull latest code
git pull

# 2. Rebuild API image only
docker-compose build hedit

# 3. Restart
docker-compose up -d hedit
```

### Updating HED Resources

HED schemas and validator are embedded in the image. To update:

```bash
# 1. Rebuild image (re-clones latest HED repos)
docker-compose build --no-cache hedit

# 2. Restart
docker-compose up -d hedit
```

Or manually:

```bash
docker exec -it hedit-api bash
cd /app/hed-schemas && git pull
# Restart container to reload
```

## Health Checks

### HEDit Container

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:38427/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

## Volume Management

### Persistent Data

No persistent volumes are required;
LLM inference runs on the Claude Platform on AWS, so no models are stored locally.

**What's NOT stored**:
- Application code (in image)
- HED resources (in image)
- Python packages (in image)
- LLM models (inference is remote)

## Security Considerations

### Production Recommendations

1. **Don't run as root**
   ```dockerfile
   RUN useradd -m -u 1000 hedit
   USER hedit
   ```

2. **Read-only filesystem**
   ```yaml
   read_only: true
   tmpfs:
     - /tmp
   ```

3. **Resource limits**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '4'
         memory: 16G
   ```

4. **Network isolation**
   ```yaml
   networks:
     - hedit-internal
   ```

## Troubleshooting

### Build Fails: npm install

**Issue**: Node.js or npm not found

**Solution**: Ensure `nodejs` and `npm` packages installed in Dockerfile

### Build Fails: git clone

**Issue**: Can't clone HED repositories

**Solution**: Check network/firewall, or use `--build-arg` to specify mirror

### Runtime: Can't find HED schemas

**Issue**: Schema directory not found

**Solution**: Verify `HED_SCHEMA_DIR` points to `/app/hed-schemas/schemas_latest_json`

### Runtime: Validator not working

**Issue**: JavaScript validator fails

**Solution**: Check `/app/hed-javascript` exists and was built during image build

## Performance

### Image Size

- Base image: ~2GB (CUDA runtime)
- With dependencies: ~3.5GB
- With HED resources: ~4GB

### Build Time

- First build: ~10-15 minutes
  - System packages: 2 min
  - HED clone + build: 5 min
  - Python packages: 3 min

- Rebuild (code change only): ~30 seconds
  - Only last layer changes
  - Docker cache used

### Runtime Performance

- Container startup: <5 seconds
- API ready: ~10 seconds
- First LLM request: a few seconds (remote call to the Claude Platform on AWS)

## Comparison: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **HED Schemas** | Host mount | Embedded in image |
| **HED Validator** | Host mount | Embedded + built |
| **Install mode** | `pip install -e .` | `pip install .` |
| **Dependencies** | External paths | Self-contained |
| **Portability** | Requires host setup | Runs anywhere |
| **Build issues** | Dependency conflicts | Clean resolution |

## Summary

The self-contained Docker architecture provides:
- ✅ Zero external dependencies
- ✅ Portable across any Docker-enabled system
- ✅ Reproducible builds
- ✅ Production-ready
- ✅ Easy deployment
- ✅ Clean dependency resolution
- ✅ Automatic HED resource inclusion

Just build, run, and go!
