# HED-BOT Deployment Guide

This document helps you choose the right deployment option for your use case.

## Quick Decision Matrix

| Use Case | Deployment Option | Documentation |
|----------|------------------|---------------|
| **Production server** (e.g., hedtools.ucsd.edu) | Production Docker | [`deploy/README.md`](deploy/README.md) |
| **Cloud hosting** (Render, Railway, Fly.io) | Production Docker | [`deploy/README.md`](deploy/README.md) |
| **Local development** | Claude Platform on AWS + Local Python | [README.md](README.md#local-development-setup) |

All deployments use Anthropic Claude large language models (LLMs) served via the Claude Platform on Amazon Web Services (AWS),
an Anthropic-operated Messages API billed through the AWS Marketplace (not Amazon Bedrock).
See [`docs/deployment/claude-platform-aws.md`](docs/deployment/claude-platform-aws.md) for credential setup.
Local Ollama-based inference is no longer supported.

## Deployment Options

### Option 1: Production Deployment (Recommended for Public Access)

**Best for:**
- Production servers (like hedtools.ucsd.edu)
- Cloud hosting platforms
- Public-facing deployments

**Features:**
- No graphics processing unit (GPU) required (uses the Claude Platform on AWS)
- Optimized 806MB Docker image
- API key authentication and audit logging
- CI/CD with GitHub Actions
- Auto-deployment with hourly updates
- Cloudflare Worker proxy (optional caching)
- OWASP Top 10 compliant security

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
- **LLM Setup**: [`docs/deployment/claude-platform-aws.md`](docs/deployment/claude-platform-aws.md) - Claude Platform on AWS credentials

**Cost:**
- Server hosting: FREE (use existing server) or $5-25/month (cloud)
- LLM API (Claude Platform on AWS): usage-based, billed through your AWS account
- Total: hosting cost plus AWS Marketplace LLM usage

---

### Option 2: Local Development

**Best for:**
- Quick testing and development
- Laptop/desktop development (no GPU needed)
- Testing against the Claude Platform on AWS

**Features:**
- No Docker required
- Fast iteration (no container rebuilds)
- Uses the Claude Platform on AWS (cloud LLM)

**Setup:**
See main [README.md](README.md#local-development-setup)
and [`docs/deployment/claude-platform-aws.md`](docs/deployment/claude-platform-aws.md)
for the required `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, and `ANTHROPIC_WORKSPACE_ID` variables.

**Cost:**
- LLM API (Claude Platform on AWS): pay-per-use through your AWS account

---

## Comparison

| Feature | Production Deploy | Local Dev |
|---------|------------------|-----------|
| **GPU Required** | No | No |
| **LLM Provider** | Claude Platform on AWS | Claude Platform on AWS |
| **Docker** | Single container | Optional |
| **Public Access** | Yes | Local only |
| **Security** | Full (API keys, audit) | Basic |
| **Auto-Updates** | Yes (CI/CD) | Manual |
| **Setup Time** | 15-30 minutes | 10-15 minutes |
| **Best For** | Production | Quick development |

---

## Migration Paths

### From Local Dev -> Production

```bash
# 1. Deploy (credentials carry over from your .env)
./deploy/deploy.sh prod

# 2. Add security configuration
python scripts/generate_api_key.py
# Add API_KEYS to .env
```

### From a Legacy OpenRouter or Ollama Setup

```bash
# 1. Update .env to use the Claude Platform on AWS
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key-here
ANTHROPIC_BASE_URL=https://aws-external-anthropic.us-east-2.api.aws
ANTHROPIC_WORKSPACE_ID=wrkspc_your_workspace_id

# 2. Remove obsolete variables
# OPENROUTER_API_KEY, LLM_PROVIDER_PREFERENCE, and the *_PROVIDER
# variables are no longer read; legacy LLM_PROVIDER values
# (openrouter, ollama) are coerced to anthropic with a warning.

# 3. Redeploy
./deploy/deploy.sh prod
```

---

## Need Help?

- **Production Deployment**: See [`deploy/README.md`](deploy/README.md)
- **Local Development**: See [README.md](README.md#local-development-setup)
- **LLM Credentials**: See [`docs/deployment/claude-platform-aws.md`](docs/deployment/claude-platform-aws.md)
- **Security Questions**: See [`deploy/SECURITY.md`](deploy/SECURITY.md)
- **Issues**: https://github.com/Annotation-Garden/HEDit/issues

---

**Last Updated**: August 18, 2026
