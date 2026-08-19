# HEDit Cloudflare Worker (Proxy Mode)

Caching proxy in front of the Python FastAPI backend,
which holds the prompts, real HED validation, and the multi-agent workflow.
LLM calls are made by the backend against Anthropic Claude
via the Claude Platform on AWS (Anthropic-operated Messages API, AWS Marketplace billing).

## Architecture

```
Frontend (Cloudflare Pages) → Worker (cache, rate limit, Turnstile) → FastAPI backend → Claude (Claude Platform on AWS)
```

**The worker provides:**
- Response caching (KV, 1 hour TTL in production)
- Per-IP rate limiting
- Cloudflare Turnstile verification for browser requests
- Header forwarding for BYOK (Bring Your Own Key) and model overrides

No LLM API keys live in the worker; Anthropic credentials are configured on the backend.

---

## Prerequisites

1. **Cloudflare Account**: [Sign up free](https://dash.cloudflare.com/sign-up)
2. **Node.js**: Install from [nodejs.org](https://nodejs.org/)
3. A running HEDit backend (see `DEPLOYMENT.md`)

---

## Quick Deployment

### Step 1: Install Wrangler CLI

```bash
npm install -g wrangler

# Login to Cloudflare
wrangler login
```

### Step 2: Create KV Namespaces

```bash
cd workers

# Create cache namespace
wrangler kv:namespace create "HED_CACHE"
# Note the ID, update wrangler.toml

# Create rate limiter namespace
wrangler kv:namespace create "RATE_LIMITER"
# Note the ID, update wrangler.toml
```

Update `wrangler.toml` with the namespace IDs.

### Step 3: Set Secrets

```bash
# Turnstile secret for browser bot protection
wrangler secret put TURNSTILE_SECRET_KEY
```

### Step 4: Deploy

```bash
wrangler deploy
```

### Step 5: Update Frontend

Edit `frontend/config.js` to point at the deployed worker URL,
then push to GitHub; Cloudflare Pages auto-deploys.

---

## BYOK and Model Headers

The worker forwards these headers to the backend:

- `X-Anthropic-Key`: user's own Anthropic API key (sk-ant-...); BYOK requests skip Turnstile
- `X-OpenRouter-Key`: legacy BYOK header, still forwarded (must carry an Anthropic key)
- `X-OpenRouter-Model`, `X-OpenRouter-Eval-Model`, `X-OpenRouter-Temperature`: model overrides
  (legacy wire names kept for compatibility); offered models are
  `claude-haiku-4-5` (default) and `claude-sonnet-5`
- `X-User-Id`: telemetry identifier

---

## Monitoring

### View Logs
```bash
wrangler tail
```

### Analytics
Visit Cloudflare Dashboard → Workers → Analytics

---

## Troubleshooting

### "Module worker script not found"
- Ensure `index.js` is in `workers/` directory
- Check `wrangler.toml` has `main = "index.js"`

### "KV namespace not found"
- Create namespaces: `wrangler kv:namespace create HED_CACHE`
- Update IDs in `wrangler.toml`

### Rate limit errors
- Clear rate limit KV: `wrangler kv:key delete --binding RATE_LIMITER "ratelimit:YOUR_IP"`

---

## Development

### Local Testing
```bash
wrangler dev

# Test locally
curl -X POST http://localhost:8787/annotate \
  -H "Content-Type: application/json" \
  -d '{"description": "Test event"}'
```

### Update Deployment
```bash
# Make changes to index.js
wrangler deploy
```

---

## Support

- [Cloudflare Workers Docs](https://developers.cloudflare.com/workers/)
- [HED Schema Documentation](https://hedtags.org/)
