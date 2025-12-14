# HED-BOT Cloudflare Workers Deployment

Fully serverless backend for HED annotation using Cloudflare Workers + OpenRouter/Cerebras.

## Architecture

```
Frontend (Cloudflare Pages) → Workers API → OpenRouter/Cerebras
```

**Benefits:**
- No backend infrastructure needed
- Auto-scales globally
- 100,000 requests/day FREE
- Built-in caching & rate limiting
- API keys secured server-side

---

## Prerequisites

1. **Cloudflare Account**: [Sign up free](https://dash.cloudflare.com/sign-up)
2. **OpenRouter API Key**: Get from [OpenRouter](https://openrouter.ai/)
3. **Node.js**: Install from [nodejs.org](https://nodejs.org/)

---

## Quick Deployment (5 minutes)

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

### Step 3: Set API Key Secret

```bash
# Set OpenRouter API key (secure, not in code)
wrangler secret put OPENROUTER_API_KEY
# Paste your OpenRouter API key when prompted
```

### Step 4: Deploy!

```bash
# Deploy to Cloudflare Workers
wrangler deploy

# You'll get a URL like: https://hed-bot-api.your-subdomain.workers.dev
```

### Step 5: Update Frontend

Edit `/home/yahya/git/hed-bot/frontend/config.js`:

```javascript
window.BACKEND_URL = 'https://hed-bot-api.your-subdomain.workers.dev';
```

Push to GitHub → Cloudflare Pages auto-deploys!

---

## Features

### 🚀 Performance
- **Cerebras model** for fast annotation (~1-2 sec)
- **Claude Sonnet** for quality evaluation
- Global CDN distribution

### 💰 Cost Optimization
- **Caching**: Identical requests served from cache (1hr TTL)
- **Rate limiting**: 10 requests/minute per IP
- **Free tier**: 100,000 requests/day

### 🔒 Security
- API keys stored as encrypted secrets
- CORS enabled for your domain only
- Rate limiting prevents abuse

---

## Configuration

Edit `workers/index.js` CONFIG section:

```javascript
const CONFIG = {
  OPENROUTER_API_URL: 'https://openrouter.ai/api/v1/chat/completions',
  DEFAULT_MODEL: 'anthropic/claude-3.5-sonnet',  // Quality model
  CEREBRAS_MODEL: 'meta-llama/llama-3.3-70b-instruct',  // Fast model
  CACHE_TTL: 3600,  // Cache duration (seconds)
  RATE_LIMIT_PER_MINUTE: 10,  // Max requests per IP
};
```

---

## API Endpoints

### `GET /health`
Health check

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "llm_available": true,
  "validator_available": true
}
```

### `POST /annotate`
Generate HED annotation

**Request:**
```json
{
  "description": "A red circle appears on the left",
  "schema_version": "8.4.0",
  "max_validation_attempts": 3,
  "run_assessment": false
}
```

**Response:**
```json
{
  "annotation": "(Sensory-event, (Visual, (Color/Red, Shape/Circle, ...))",
  "is_valid": true,
  "is_faithful": true,
  "is_complete": true,
  "validation_attempts": 1,
  "validation_errors": [],
  "validation_warnings": [],
  "evaluation_feedback": "...",
  "assessment_feedback": "...",
  "status": "success"
}
```

---

## Monitoring

### View Logs
```bash
wrangler tail
```

### Analytics
Visit Cloudflare Dashboard → Workers → hed-bot-api → Analytics

### Cache Stats
```bash
# Check cache hits/misses
wrangler kv:namespace list --binding HED_CACHE
```

---

## Costs

### Free Tier (plenty for most use)
- 100,000 requests/day
- 10ms CPU time per request
- KV reads/writes included

### Paid (if needed)
- $0.50 per million requests
- $0.50 per million KV reads

### OpenRouter API
- Cerebras Llama 3.3 70B: ~$0.0001/request (ultra-cheap!)
- Claude Sonnet: ~$0.003/request

**Estimated monthly cost for 10,000 annotations:**
- Workers: FREE (under 100k/day limit)
- OpenRouter: ~$2-5 (mostly Cerebras, some Claude)

**Total: ~$2-5/month** for production use! 🎉

---

## Advanced Features

### Custom Domain

1. Add custom domain in Cloudflare Dashboard
2. Update `wrangler.toml`:
```toml
routes = [
  { pattern = "api.your-domain.com/*", zone_name = "your-domain.com" }
]
```
3. Deploy: `wrangler deploy`

### Increase Rate Limits

Edit `CONFIG.RATE_LIMIT_PER_MINUTE` in `index.js`.

### Add Authentication

Add API key check:
```javascript
if (request.headers.get('X-API-Key') !== env.API_KEY) {
  return new Response('Unauthorized', { status: 401 });
}
```

---

## Troubleshooting

### "Module worker script not found"
- Ensure `index.js` is in `workers/` directory
- Check `wrangler.toml` has `main = "index.js"`

### "KV namespace not found"
- Create namespaces: `wrangler kv:namespace create HED_CACHE`
- Update IDs in `wrangler.toml`

### "OpenRouter API error"
- Check API key: `wrangler secret list`
- Set if missing: `wrangler secret put OPENROUTER_API_KEY`

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
# Changes live instantly!
```

---

## Next Steps

1. Deploy Workers ✅
2. Deploy Frontend to Cloudflare Pages
3. Test end-to-end
4. Monitor usage & costs
5. Add custom domain (optional)

---

## Support

- [Cloudflare Workers Docs](https://developers.cloudflare.com/workers/)
- [OpenRouter API Docs](https://openrouter.ai/docs)
- [HED Schema Documentation](https://hedtags.org/)
