# Docker Deployment & Code Updates

This guide explains how to deploy code changes when running HED-BOT with Docker.

## The Problem

**`docker-compose restart` does NOT reload your code!**

When you run `docker-compose restart`:
- ✗ Does NOT pull new code
- ✗ Does NOT rebuild the image
- ✓ Only restarts the existing container with old code

## The Solution

Use the deployment script to properly rebuild and deploy:

```bash
./scripts/deploy.sh --force
```

## Quick Reference

```bash
# After pulling code changes
./scripts/deploy.sh --force

# Check version
curl http://localhost:38427/version

# View logs
docker-compose logs -f hed-bot
```

## Deployment Script Usage

### Basic Usage

```bash
# Normal deployment (uses cache)
./scripts/deploy.sh

# Force rebuild (recommended after code changes)
./scripts/deploy.sh --force

# Help
./scripts/deploy.sh --help
```

### What It Does

1. Stops existing containers
2. Rebuilds the image with your latest code
3. Starts services
4. Waits for health checks
5. Shows deployment summary with version

### When to Use `--force`

Use `--force` flag (no cache) for:
- ✓ After pulling code changes
- ✓ After version bumps
- ✓ Dependency changes (`pyproject.toml`)
- ✓ When you want to ensure clean build

Use normal mode (with cache) for:
- Minor config changes
- Faster rebuilds during development

## Manual Deployment

If you prefer manual steps:

```bash
# 1. Stop containers
docker-compose down

# 2. Rebuild hed-bot image (no cache)
docker-compose build --no-cache hed-bot

# 3. Start services
docker-compose up -d

# 4. Check logs
docker-compose logs -f hed-bot

# 5. Verify version
curl http://localhost:38427/version
```

## Common Workflow

### After Merging a PR

```bash
# 1. Pull latest code
git checkout main
git pull origin main

# 2. Deploy
./scripts/deploy.sh --force

# 3. Verify version updated
curl http://localhost:38427/version
# Check frontend at http://localhost or your domain
```

### After Version Bump

```bash
# 1. Bump version
python scripts/bump_version.py patch

# 2. Push
git push origin main
git push origin v0.3.1-alpha  # your version

# 3. Deploy
./scripts/deploy.sh --force

# 4. Verify in browser
# Frontend footer should show new version
```

## Useful Docker Commands

```bash
# View logs (follow mode)
docker-compose logs -f hed-bot

# View logs (last 100 lines)
docker-compose logs --tail=100 hed-bot

# Check container status
docker-compose ps

# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes Ollama models!)
docker-compose down -v

# Execute command in container
docker-compose exec hed-bot bash

# View resource usage
docker stats

# Restart just one service
docker-compose restart hed-bot
# Note: This WON'T pick up code changes!
```

## Troubleshooting

### Version Shows Old Value

**Problem**: Frontend still shows old version after deployment

**Steps**:
1. Force rebuild: `./scripts/deploy.sh --force`
2. Clear browser cache (Ctrl+Shift+R or Cmd+Shift+R)
3. Verify API: `curl http://localhost:38427/version`
4. Check logs: `docker-compose logs hed-bot`

### Container Won't Start

**Problem**: Container exits immediately

**Steps**:
```bash
# Check logs for errors
docker-compose logs hed-bot

# Common causes:
# - Port 38427 already in use
# - Missing environment variables
# - Python import errors

# Try clean rebuild
docker-compose down
docker-compose build --no-cache hed-bot
docker-compose up -d
```

### Port Already in Use

**Problem**: `address already in use: 38427`

**Solution**:
```bash
# Find process using port
sudo lsof -i :38427

# Kill it
sudo kill -9 <PID>

# Or change port in docker-compose.yml
```

### Image Build Fails

**Problem**: Build errors during `docker-compose build`

**Steps**:
1. Check error message in output
2. Common issues:
   - Network problems (HED repo clone)
   - Node.js build failures
   - Python dependency conflicts
3. Try: `docker-compose build --no-cache --pull hed-bot`

## Development Mode

### Option 1: Volume Mount (Auto-reload)

Edit `docker-compose.yml`:

```yaml
services:
  hed-bot:
    # ... existing config ...
    volumes:
      - ./src:/app/src  # Mount source code
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 38427 --reload
```

Then:
```bash
docker-compose up -d
# Now code changes auto-reload!
```

### Option 2: Run Locally (Recommended for Development)

```bash
# Activate conda environment
conda activate hed-bot

# Set environment variables (if using OpenRouter)
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=your-key

# Run with auto-reload
uvicorn src.api.main:app --host 0.0.0.0 --port 38427 --reload

# Code changes now auto-reload!
```

## Production Deployment

### Build Tagged Image

```bash
# Get current version
VERSION=$(python scripts/bump_version.py --current | awk '{print $3}')

# Build with version tag
docker build -t hed-bot:$VERSION -t hed-bot:latest .
```

### Push to Registry (Optional)

```bash
# Tag for your registry
docker tag hed-bot:$VERSION your-registry.com/hed-bot:$VERSION

# Push
docker push your-registry.com/hed-bot:$VERSION
docker push your-registry.com/hed-bot:latest
```

## Understanding Docker Caching

Docker uses layer caching to speed up builds:

- **With cache** (`docker-compose build`):
  - Reuses unchanged layers
  - Faster builds
  - May miss some changes

- **Without cache** (`docker-compose build --no-cache`):
  - Rebuilds everything from scratch
  - Slower but guaranteed fresh
  - Recommended after code changes

## Summary

**Key Takeaways**:

1. ✗ **DON'T** use `docker-compose restart` for code changes
2. ✓ **DO** use `./scripts/deploy.sh --force` after code changes
3. ✓ **DO** clear browser cache to see frontend changes
4. ✓ **DO** check logs if something goes wrong

**Quick Commands**:
```bash
# Deploy code changes
./scripts/deploy.sh --force

# Check version
curl http://localhost:38427/version

# View logs
docker-compose logs -f hed-bot

# Stop everything
docker-compose down
```

## Related Documentation

- `VERSION.md` - Version management and bumping
- `DEPLOYMENT.md` - Cloudflare Pages & Tunnel deployment
- `README.md` - General setup and usage
