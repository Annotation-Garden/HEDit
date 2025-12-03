# Docker Quick Start (Local GPU Development)

> **Note**: This guide is for local development with GPU and Ollama. For production deployment to servers or cloud platforms, see **[DEPLOYMENT.md](../../DEPLOYMENT.md)** and **[deploy/README.md](../../deploy/README.md)**.

## TL;DR

```bash
# Clone and start (everything included!)
git clone <repo> && cd hed-bot
docker-compose up -d

# Monitor first start (~10-20 min for model download)
docker-compose logs -f

# Verify when ready
curl http://localhost:38427/health

# Open frontend
open frontend/index.html
```

## What's Different?

### Before (Manual Setup)
```bash
# Required external HED repositories
/Users/yahya/Documents/git/HED/hed-schemas      ❌ Host dependency
/Users/yahya/Documents/git/HED/hed-javascript   ❌ Host dependency

# Volume mounts needed
docker-compose.yml:
  volumes:
    - ./src:/app/src                            ❌ Editable install
    - /path/to/hed-schemas:/schemas             ❌ External mount

# Manual steps
npm install in hed-javascript                   ❌ Host setup
pip install -e . (editable mode)                ❌ Dependency issues
```

### After (Self-Contained)
```bash
# Everything in the container!
/app/hed-schemas     ✅ Cloned during build
/app/hed-javascript  ✅ Cloned & built during build

# No volume mounts
docker-compose.yml:
  # No src mount needed!                        ✅ Clean install
  # No external HED mounts!                     ✅ Self-contained

# Zero manual steps
Just: docker-compose up -d                      ✅ One command
```

## What's Included in the Image?

1. **Base System**
   - Ubuntu 22.04 with CUDA 12.2
   - Python 3.11
   - Node.js 18+

2. **HED Resources** (Auto-cloned from GitHub)
   - HED schemas (latest) → `/app/hed-schemas`
   - HED JavaScript validator → `/app/hed-javascript` (built)

3. **Application**
   - HED-BOT source code
   - All Python dependencies
   - Installed as package (not editable)

4. **Environment**
   - `HED_SCHEMA_DIR=/app/hed-schemas/schemas_latest_json`
   - `HED_VALIDATOR_PATH=/app/hed-javascript`
   - `USE_JS_VALIDATOR=true`

## Build Process

```dockerfile
# 1. Install system packages
apt-get install python3.11 nodejs npm git curl

# 2. Clone HED resources (self-contained!)
git clone hed-schemas /app/hed-schemas
git clone hed-javascript /app/hed-javascript

# 3. Build validator
cd /app/hed-javascript && npm install && npm run build

# 4. Install application
pip install .  # Not -e (editable)!
```

## First Deployment

```bash
# 1. Build images (includes HED resources)
docker-compose build

# Time: ~10-15 minutes
# - System packages: 2 min
# - Clone HED repos: 1 min
# - Build validator: 4 min
# - Python packages: 3 min

# 2. Start containers (auto-pulls gpt-oss:20b)
docker-compose up -d

# Time: 10-20 minutes (first time only)
# - Model download: gpt-oss:20b (~12GB)

# 3. Monitor progress
docker-compose logs -f

# 4. Verify
curl http://localhost:38427/health
```

## Updating

### Update Application Code
```bash
git pull
docker-compose build hed-bot
docker-compose up -d hed-bot
# Time: ~30 seconds (Docker cache)
```

### Update HED Resources
```bash
# Rebuild image (re-clones latest HED repos)
docker-compose build --no-cache hed-bot
docker-compose up -d hed-bot
# Time: ~10 minutes (full rebuild)
```

## Troubleshooting

### Build Error: "pip install failed"
**Before**: Editable install (`pip install -e .`) caused dependency conflicts
**After**: Regular install (`pip install .`) - clean resolution ✅

### Build Error: "Can't find hed-schemas"
**Before**: Needed host directory `/Users/yahya/.../hed-schemas`
**After**: Cloned into image during build ✅

### Runtime Error: "Validator not found"
**Before**: Needed host directory with `npm run build`
**After**: Built into image automatically ✅

### Container Unhealthy
```bash
# Check logs
docker-compose logs hed-bot

# Check HED resources exist
docker exec -it hed-bot-api ls -la /app/hed-schemas
docker exec -it hed-bot-api ls -la /app/hed-javascript

# Verify validator built
docker exec -it hed-bot-api ls -la /app/hed-javascript/dist
```

## Benefits

### For Development
- ✅ No external HED repository setup needed
- ✅ Clean dependency resolution
- ✅ Consistent environment across machines
- ✅ Fast rebuilds with Docker cache

### For Production
- ✅ Fully portable container
- ✅ Zero host dependencies (except GPU)
- ✅ Reproducible deployments
- ✅ Easy updates and rollbacks

### For Deployment
- ✅ Single `docker-compose up` command
- ✅ No manual setup steps
- ✅ Works on any Docker-enabled system
- ✅ Self-healing with health checks

## Architecture

See [docs/DOCKER_ARCHITECTURE.md](docs/DOCKER_ARCHITECTURE.md) for detailed architecture documentation.

## Next Steps

1. **Test Locally**: `docker-compose up -d`
2. **Verify Health**: `curl http://localhost:38427/health`
3. **Test Annotation**: Use frontend or API
4. **Deploy to Server**: Same commands work everywhere!

## Summary

The new self-contained Docker architecture means:
- **Zero setup**: No HED repos needed on host
- **One command**: `docker-compose up -d`
- **Fully portable**: Works anywhere Docker runs
- **Production-ready**: Clean, reproducible, self-healing

Just build and run! 🚀
