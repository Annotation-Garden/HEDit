# HED-BOT Deployment Guide

This document helps you choose the right deployment option for your use case.

## Quick Decision Matrix

| Use Case | Deployment Option | Documentation |
|----------|------------------|---------------|
| **Production server** (e.g., hedtools.ucsd.edu) | Production Docker | [`deploy/README.md`](deploy/README.md) |
| **Cloud hosting** (Render, Railway, Fly.io) | Production Docker | [`deploy/README.md`](deploy/README.md) |
| **Local GPU development** | Docker Compose + Ollama | [`docs/deployment/docker-quickstart.md`](docs/deployment/docker-quickstart.md) |
| **Local development** (no GPU) | OpenRouter + Local Python | [README.md](README.md#local-development-setup) |

## Deployment Options

### Option 1: Production Deployment (Recommended for Public Access)

**Best for:**
- Production servers (like hedtools.ucsd.edu)
- Cloud hosting platforms
- Public-facing deployments
- Teams without local GPU

**Features:**
- ✅ No GPU required (uses OpenRouter API)
- ✅ Optimized 806MB Docker image
- ✅ API key authentication & audit logging
- ✅ CI/CD with GitHub Actions
- ✅ Auto-deployment with hourly updates
- ✅ Cloudflare Worker proxy (optional caching)
- ✅ OWASP Top 10 compliant security

**Setup:**
```bash
# See complete guide:
cat deploy/README.md

# Quick start:
python scripts/generate_api_key.py
./deploy/deploy.sh prod
```

**Documentation:**
- **Main Guide**: [`deploy/README.md`](deploy/README.md) - Complete deployment documentation
- **Security**: [`deploy/SECURITY.md`](deploy/SECURITY.md) - Audit-ready security guide
- **Architecture**: [`deploy/DEPLOYMENT_ARCHITECTURE.md`](deploy/DEPLOYMENT_ARCHITECTURE.md) - CORS and reverse proxy setup

**Cost:**
- Server hosting: FREE (use existing server) or $5-25/month (cloud)
- LLM API (OpenRouter): ~$2-5/month for moderate usage
- Total: ~$2-30/month depending on hosting choice

---

### Option 2: Local GPU Development

**Best for:**
- Local development and testing
- Privacy-sensitive work (offline LLM)
- GPU available (NVIDIA RTX 3090/4090 or better)
- No ongoing API costs desired

**Features:**
- ✅ Fully offline operation
- ✅ No API costs after setup
- ✅ Complete privacy (no external API calls)
- ✅ Self-contained Docker setup
- ✅ Auto-downloads Ollama models

**Requirements:**
- NVIDIA GPU with 24GB+ VRAM (RTX 3090/4090)
- NVIDIA Container Toolkit
- Docker Compose

**Setup:**
```bash
# See complete guide:
cat docs/deployment/docker-quickstart.md

# Quick start:
docker-compose up -d
docker-compose logs -f  # Wait for model download (~10-20 min)
```

**Documentation:**
- **Quick Start**: [`docs/deployment/docker-quickstart.md`](docs/deployment/docker-quickstart.md) - One-command setup
- **Architecture**: [`docs/deployment/docker-architecture.md`](docs/deployment/docker-architecture.md) - Technical details
- **Ollama Config**: [`docker/README.md`](docker/README.md) - Model configuration

**Cost:**
- One-time: GPU hardware (~$1200-1600 for RTX 4090)
- Electricity: ~$10-30/month (GPU power consumption)
- Total: FREE after hardware investment

---

### Option 3: Local Development (No GPU)

**Best for:**
- Quick testing and development
- Laptop/desktop without GPU
- OpenRouter API testing

**Features:**
- ✅ No Docker required
- ✅ Fast iteration (no container rebuilds)
- ✅ Uses OpenRouter API (cloud LLM)
- ✅ Conda environment management

**Setup:**
See main [README.md](README.md#local-development-setup)

**Cost:**
- LLM API (OpenRouter): Pay-per-use (~$0.10-0.50 per session)

---

## Comparison

| Feature | Production Deploy | Local GPU | Local Dev (no GPU) |
|---------|------------------|-----------|-------------------|
| **GPU Required** | ❌ No | ✅ Yes (24GB+) | ❌ No |
| **LLM Provider** | OpenRouter (cloud) | Ollama (local) | OpenRouter (cloud) |
| **Docker** | ✅ Single container | ✅ Compose (2 containers) | ❌ Optional |
| **Public Access** | ✅ Yes | ❌ Local only | ❌ Local only |
| **Security** | ✅ Full (API keys, audit) | ⚠️ Basic | ⚠️ Basic |
| **Auto-Updates** | ✅ Yes (CI/CD) | ❌ Manual | ❌ Manual |
| **Setup Time** | 15-30 minutes | 30-45 minutes (+ model download) | 10-15 minutes |
| **Monthly Cost** | $2-30 | $10-30 (electricity) | $0.10-0.50 per session |
| **Best For** | Production | Privacy, offline | Quick development |

---

## Migration Paths

### From Local GPU → Production

```bash
# 1. Set up production deployment
cd deploy/
./deploy.sh prod

# 2. Update .env to use OpenRouter instead of Ollama
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-key-here

# 3. Stop local GPU setup (optional)
docker-compose down
```

### From Local Dev → Production

```bash
# 1. Already using OpenRouter - just deploy
./deploy/deploy.sh prod

# 2. Add security configuration
python scripts/generate_api_key.py
# Add API_KEYS to .env
```

---

## Need Help?

- **Production Deployment**: See [`deploy/README.md`](deploy/README.md)
- **Local GPU Setup**: See [`docs/deployment/docker-quickstart.md`](docs/deployment/docker-quickstart.md)
- **Local Development**: See [README.md](README.md#local-development-setup)
- **Security Questions**: See [`deploy/SECURITY.md`](deploy/SECURITY.md)
- **Issues**: https://github.com/neuromechanist/hed-bot/issues

---

**Last Updated**: December 2, 2025
