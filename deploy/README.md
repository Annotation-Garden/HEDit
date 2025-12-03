# HED-BOT Deployment Guide

This directory contains deployment scripts and configuration for running HED-BOT in production.

## Table of Contents

- [Quick Start](#quick-start)
- [Security Setup](#security-setup)
- [Deployment Options](#deployment-options)
- [Automated Deployment](#automated-deployment)
- [Manual Deployment](#manual-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Docker installed on server
- Git access to repository
- Port 38427 available

### Deploy Production

```bash
# Clone repository
git clone https://github.com/neuromechanist/hed-bot.git
cd hed-bot

# Generate API key for authentication
python scripts/generate_api_key.py

# Create .env file with your configuration
cp .env.example .env
# Edit .env with your API keys (both OPENROUTER_API_KEY and API_KEYS)

# Deploy
./deploy/deploy.sh prod
```

**Important:** See [Security Setup](#security-setup) for detailed security configuration before deploying to production.

---

## Security Setup

HED-BOT implements comprehensive security features for production deployment:

- **API Key Authentication**: Protects endpoints from unauthorized access
- **Audit Logging**: Complete request/response trail for compliance
- **CORS Validation**: Restricts origins to approved frontends only
- **Security Headers**: Protects against common web attacks

### Quick Security Setup

1. **Generate API Key:**
   ```bash
   python scripts/generate_api_key.py
   ```

2. **Add to .env file:**
   ```bash
   # API Authentication
   API_KEYS=<your_generated_key>
   REQUIRE_API_AUTH=true

   # Audit Logging
   ENABLE_AUDIT_LOG=true
   AUDIT_LOG_FILE=/var/log/hed-bot/audit.log

   # CORS (optional extra origins)
   # EXTRA_CORS_ORIGINS=https://staging.hed-bot.pages.dev
   ```

3. **Use API key in requests:**
   ```bash
   curl -H "X-API-Key: your_key_here" \
        https://hedtools.ucsd.edu/hed-bot-api/annotate
   ```

### Protected vs Public Endpoints

**Protected (require API key):**
- `POST /annotate` - Generate annotations
- `POST /annotate-from-image` - Image annotations
- `POST /validate` - Validate HED strings

**Public (no authentication):**
- `GET /health` - Health checks
- `GET /version` - Version info
- `GET /` - API documentation

### Complete Security Documentation

For comprehensive security information including:
- OWASP Top 10 compliance
- Audit log formats and retention
- Incident response procedures
- Security checklist for auditors

**See:** [`SECURITY.md`](SECURITY.md) for complete security documentation.

**See Also:** [`DEPLOYMENT_ARCHITECTURE.md`](DEPLOYMENT_ARCHITECTURE.md) for architecture details and CORS configuration.

---

## Deployment Options

### Option 1: Automated Deployment (Recommended)

Automatically checks for and deploys new releases every hour.

**Setup:**

```bash
# Test the auto-update script
./deploy/auto-update.sh --check-only

# Add to crontab for hourly checks
crontab -e

# Add this line:
0 * * * * /path/to/hed-bot/deploy/auto-update.sh >> /var/log/hed-bot-update.log 2>&1
```

**How it works:**
1. GitHub Actions builds Docker image on every push to `main`
2. Image is pushed to GitHub Container Registry (GHCR)
3. Cron job checks for new images hourly
4. Automatically pulls and deploys new version if available
5. Logs all updates to `/var/log/hed-bot-update.log`

### Option 2: Manual Deployment

Deploy manually when needed.

**Production:**
```bash
./deploy/deploy.sh prod
```

**Development:**
```bash
./deploy/deploy.sh dev
```

---

## Automated Deployment

### GitHub Actions Workflow

When code is pushed to `main` branch or a version tag is created:

1. **Build**: Docker image is built from `deploy/Dockerfile`
2. **Test**: (Future) Run tests against the image
3. **Push**: Image is pushed to `ghcr.io/neuromechanist/hed-bot`
4. **Tag**: Images are tagged with:
   - `latest` (from main branch)
   - `main` (latest main branch)
   - `v1.2.3` (from version tags)
   - `sha-abc123` (commit hash)

### Server-Side Auto-Update

The `auto-update.sh` script provides automated deployment:

#### Features

- **Automatic Updates**: Checks for new Docker images
- **Safe Deployment**: Uses locking to prevent concurrent updates
- **Rollback Ready**: Keeps previous image for quick rollback
- **Logging**: Comprehensive logs for troubleshooting
- **Cleanup**: Removes old dangling images

#### Usage

```bash
# Check for updates without deploying
./deploy/auto-update.sh --check-only

# Force update even if no new image
./deploy/auto-update.sh --force

# Update dev environment
./deploy/auto-update.sh --env dev

# Run from cron (recommended)
0 * * * * /path/to/deploy/auto-update.sh >> /var/log/hed-bot-update.log 2>&1
```

#### Configuration

Edit the script to customize:

```bash
# Image registry
REGISTRY_IMAGE="ghcr.io/neuromechanist/hed-bot:latest"

# Log file location
LOG_FILE="/var/log/hed-bot-update.log"

# Lock file location
LOCK_FILE="/tmp/hed-bot-update.lock"
```

---

## Manual Deployment

### deploy.sh Script

#### Syntax

```bash
./deploy.sh [environment] [bind_address]
```

#### Parameters

- **environment**: `prod` (default) or `dev`
- **bind_address**: IP to bind to (default: `127.0.0.1`)

#### Examples

```bash
# Production deployment (127.0.0.1:38427)
./deploy.sh prod

# Development deployment (127.0.0.1:38428)
./deploy.sh dev

# Bind to all interfaces (not recommended for production)
./deploy.sh prod 0.0.0.0
```

#### What it does

1. Builds Docker image from `deploy/Dockerfile`
2. Stops existing container (if running)
3. Starts new container with:
   - Port mapping (host:38427 → container:38427)
   - Environment variables from `.env`
   - Auto-restart policy
   - Health checks enabled
4. Shows logs and verifies health
5. Displays useful management commands

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Authentication (REQUIRED for production)
API_KEYS=your_generated_key_here
REQUIRE_API_AUTH=true

# Audit Logging (recommended for production)
ENABLE_AUDIT_LOG=true
AUDIT_LOG_FILE=/var/log/hed-bot/audit.log

# CORS Configuration (optional extra origins)
# EXTRA_CORS_ORIGINS=https://staging.hed-bot.pages.dev,https://dev.hed-bot.pages.dev

# LLM Configuration (Cerebras + OpenRouter for ultra-fast inference)
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_key_here
LLM_PROVIDER_PREFERENCE=Cerebras
LLM_TEMPERATURE=0.1

# Model configuration (Cerebras-optimized defaults)
ANNOTATION_MODEL=openai/gpt-oss-120b
EVALUATION_MODEL=qwen/qwen3-235b-a22b-2507
ASSESSMENT_MODEL=openai/gpt-oss-120b
FEEDBACK_MODEL=openai/gpt-oss-120b

# Optional: HED Schema and Validator paths (if not using defaults)
# HED_SCHEMA_DIR=/path/to/hed-schemas
# HED_VALIDATOR_PATH=/path/to/hed-javascript
# USE_JS_VALIDATOR=true
```

**Security Note:** Generate API keys using `python scripts/generate_api_key.py`. Never commit `.env` to Git.

### Reverse Proxy Configuration

#### Apache (hedtools.ucsd.edu)

Add to your Apache virtual host configuration:

```apache
# HED-BOT API Backend
ProxyPass /hed-bot-api/ http://localhost:38427/
ProxyPassReverse /hed-bot-api/ http://localhost:38427/
```

Reload Apache:
```bash
sudo apache2ctl configtest
sudo systemctl reload apache2
```

#### Nginx (Alternative)

```nginx
location /hed-bot-api/ {
    proxy_pass http://127.0.0.1:38427/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_connect_timeout 120s;
    proxy_send_timeout 120s;
    proxy_read_timeout 120s;
}
```

Reload Nginx:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## Monitoring

### Container Status

```bash
# Check if container is running
docker ps | grep hed-bot

# Check container health
docker inspect --format='{{.State.Health.Status}}' hed-bot

# View resource usage
docker stats hed-bot
```

### Logs

```bash
# View real-time logs
docker logs -f hed-bot

# View last 100 lines
docker logs --tail 100 hed-bot

# View logs with timestamps
docker logs -t hed-bot

# Auto-update logs
tail -f /var/log/hed-bot-update.log
```

### Health Checks

```bash
# Manual health check
curl http://localhost:38427/health

# Through reverse proxy
curl https://your-domain.com/hed-bot/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.4.2-alpha",
  "llm_available": true,
  "validator_available": true
}
```

---

## Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker logs hed-bot
```

**Common issues:**
- Missing `.env` file → Copy from `.env.example`
- Invalid API key → Check OPENROUTER_API_KEY
- Port already in use → `sudo lsof -i :38427`

### Auto-Update Not Working

**Check cron job:**
```bash
crontab -l
```

**Check permissions:**
```bash
chmod +x deploy/auto-update.sh
```

**Check logs:**
```bash
tail -f /var/log/hed-bot-update.log
```

**Test manually:**
```bash
./deploy/auto-update.sh --check-only
```

### Container Running But Not Responding

**Check health:**
```bash
docker inspect --format='{{.State.Health}}' hed-bot
```

**Check network:**
```bash
curl http://127.0.0.1:38427/health
```

**Restart container:**
```bash
docker restart hed-bot
```

### Rollback to Previous Version

**If auto-update caused issues:**

```bash
# Stop current container
docker stop hed-bot
docker rm hed-bot

# Find previous image
docker images | grep hed-bot

# Run previous image
docker run -d \
  --name hed-bot \
  --restart unless-stopped \
  -p 127.0.0.1:38427:38427 \
  --env-file .env \
  hed-bot:previous-tag
```

### Docker Image Not Pulling

**Login to GitHub Container Registry:**
```bash
# Create personal access token with read:packages scope
# https://github.com/settings/tokens

echo YOUR_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
```

**Manually pull image:**
```bash
docker pull ghcr.io/neuromechanist/hed-bot:latest
```

---

## Advanced Topics

### Custom Deployment Schedule

Modify cron schedule for different update frequencies:

```bash
# Every 15 minutes
*/15 * * * * /path/to/auto-update.sh >> /var/log/hed-bot-update.log 2>&1

# Every 6 hours
0 */6 * * * /path/to/auto-update.sh >> /var/log/hed-bot-update.log 2>&1

# Once daily at 2 AM
0 2 * * * /path/to/auto-update.sh >> /var/log/hed-bot-update.log 2>&1

# Weekdays at 6 AM
0 6 * * 1-5 /path/to/auto-update.sh >> /var/log/hed-bot-update.log 2>&1
```

### Notification Integration

Add notification to `auto-update.sh`:

```bash
send_notification() {
    MESSAGE="$1"

    # Slack webhook
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"$MESSAGE\"}" \
        YOUR_SLACK_WEBHOOK_URL

    # Email
    echo "$MESSAGE" | mail -s "HED-BOT Update" admin@example.com
}
```

### Blue-Green Deployment

Run both old and new versions simultaneously:

```bash
# Deploy new version on alternate port
./deploy.sh dev  # Runs on port 38428

# Test new version
curl http://localhost:38428/health

# Switch reverse proxy to new version
# Update Nginx configuration

# Remove old version
docker stop hed-bot
docker rm hed-bot
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `Dockerfile` | Container build configuration |
| `deploy.sh` | Manual deployment script |
| `auto-update.sh` | Automated update script |
| `nginx-hedtools.conf` | Nginx reverse proxy configuration |
| `SECURITY.md` | Security documentation for auditors |
| `DEPLOYMENT_ARCHITECTURE.md` | Architecture and CORS setup guide |
| `README.md` | This documentation |

---

## Support

- **Issues**: https://github.com/neuromechanist/hed-bot/issues
- **Documentation**: https://github.com/neuromechanist/hed-bot/tree/main/docs

---

**Last Updated**: December 2, 2025
