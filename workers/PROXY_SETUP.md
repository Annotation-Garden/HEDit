# HED-BOT Cloudflare Worker Proxy Setup

The Cloudflare Worker acts as a **caching proxy** to the Python FastAPI backend on hedtools.ucsd.edu, providing edge caching, rate limiting, and CORS validation.

## Architecture

```
User → Cloudflare Worker → hedtools.ucsd.edu/hed-bot → Docker Container
        (caching, rate limiting,       (Nginx reverse proxy)    (FastAPI backend)
         CORS validation)
```

## Features

### Caching
- **1 hour TTL** for successful annotations
- **Reduces API costs** and improves response time
- **Cache key**: `hed:{schema_version}:{description_hash}`
- Only caches valid annotations to avoid caching errors

### Rate Limiting
- **20 requests per minute** per IP address
- Prevents abuse and protects backend
- Uses Cloudflare KV for distributed rate limiting

### CORS Protection
- **Restricted to**: `https://hed-bot.pages.dev` (production frontend)
- **Development**: Also allows `http://localhost:*` for local testing
- Blocks all other origins

### Request Timeout
- **2 minutes** for annotation workflows
- **30 seconds** for validation requests
- Prevents hanging requests

## Setup Steps

### 1. Deploy Backend to hedtools.ucsd.edu

Follow the deployment guide in `deploy/README.md`:

```bash
# On hedtools.ucsd.edu server
cd /path/to/hed-bot

# Generate API key for worker
python scripts/generate_api_key.py

# Add to .env
echo "API_KEYS=<generated_key>" >> .env

# Deploy container
./deploy/deploy.sh prod

# Configure Nginx (already done if following deployment guide)
sudo systemctl reload nginx
```

### 2. Configure Worker Secrets

Set the backend API key as a secret:

```bash
cd /home/yahya/git/hed-bot/workers

# Set API key (use the key generated in step 1)
echo "your-api-key-here" | npx wrangler secret put BACKEND_API_KEY
```

### 3. Deploy Worker

```bash
# Deploy to Cloudflare
npx wrangler deploy
```

You should see output like:
```
Published hed-bot-api (0.42 sec)
  https://hed-bot-api.shirazi-10f.workers.dev
```

### 4. Test the Setup

#### Test health endpoint:
```bash
curl https://hed-bot-api.shirazi-10f.workers.dev/health | jq .
```

Expected output:
```json
{
  "status": "healthy",
  "proxy": "operational",
  "backend": {
    "status": "healthy",
    "version": "0.4.0-alpha",
    "llm_available": true,
    "validator_available": true
  },
  "backend_url": "https://hedtools.ucsd.edu/hed-bot"
}
```

#### Test annotation:
```bash
curl -X POST https://hed-bot-api.shirazi-10f.workers.dev/annotate \
  -H "Content-Type: application/json" \
  -H "Origin: https://hed-bot.pages.dev" \
  -d '{
    "description": "A red circle appears on the left side of the screen",
    "schema_version": "8.4.0",
    "max_validation_attempts": 3
  }' | jq .
```

Expected output:
```json
{
  "annotation": "Sensory-event, Visual-presentation, ...",
  "is_valid": true,
  "is_faithful": true,
  "validation_attempts": 1,
  "cached": false
}
```

#### Test caching:
```bash
# Run the same request again - should return cached result
curl -X POST https://hed-bot-api.shirazi-10f.workers.dev/annotate \
  -H "Content-Type: application/json" \
  -H "Origin: https://hed-bot.pages.dev" \
  -d '{
    "description": "A red circle appears on the left side of the screen",
    "schema_version": "8.4.0"
  }' | jq '.cached'
# Should return: true
```

### 5. Update Frontend

Update your frontend to use the worker URL:

```javascript
// Frontend configuration
const API_URL = 'https://hed-bot-api.shirazi-10f.workers.dev';

// All requests automatically include proper CORS headers
fetch(`${API_URL}/annotate`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    description: 'person sees red circle',
  }),
});
```

## Configuration

### Environment Variables (wrangler.toml)

```toml
[vars]
ENVIRONMENT = "production"
BACKEND_URL = "https://hedtools.ucsd.edu/hed-bot"
```

### Secrets (set via wrangler CLI)

```bash
# Backend API key for authentication
npx wrangler secret put BACKEND_API_KEY
```

### Worker Configuration (index.js)

```javascript
const CONFIG = {
  CACHE_TTL: 3600,                    // 1 hour cache
  RATE_LIMIT_PER_MINUTE: 20,          // 20 requests per minute per IP
  REQUEST_TIMEOUT: 120000,            // 2 minutes for workflows
  ALLOWED_ORIGIN: 'https://hed-bot.pages.dev',  // Production frontend only
};
```

## Troubleshooting

### "Backend not configured" error
```bash
# Check if BACKEND_URL is set in wrangler.toml
cat wrangler.toml | grep BACKEND_URL
```

### "Backend unreachable" error
```bash
# Test backend directly
curl https://hedtools.ucsd.edu/hed-bot/health

# Check Docker container
docker ps | grep hed-bot

# Check Nginx
sudo systemctl status nginx
```

### "401 Unauthorized" from backend
```bash
# Verify API key secret is set
npx wrangler secret list

# Should show BACKEND_API_KEY in the list
# If not, set it:
echo "your-api-key" | npx wrangler secret put BACKEND_API_KEY
```

### "Rate limit exceeded"
```bash
# Check rate limiter KV namespace
npx wrangler kv:key list --namespace-id=8deb8c02177349afb1d1fe2f7d4079f4

# Clear rate limit for specific IP (if needed)
npx wrangler kv:key delete "ratelimit:1.2.3.4" \
  --namespace-id=8deb8c02177349afb1d1fe2f7d4079f4
```

### CORS errors in browser
```bash
# Verify origin is allowed
# Production: https://hed-bot.pages.dev
# Development: http://localhost:*

# Check browser console for actual origin being sent
# Update CONFIG.ALLOWED_ORIGIN if needed
```

## Monitoring

### View Worker Logs

```bash
# Tail logs in real-time
npx wrangler tail

# Or view in Cloudflare dashboard:
# Workers & Pages → hed-bot-api → Logs
```

### Check Cache Hit Rate

Monitor the `cached: true/false` field in responses to see caching effectiveness.

### Check Rate Limiting

View KV namespace metrics in Cloudflare dashboard to see rate limiting activity.

## Architecture Benefits

### Why Use Worker Proxy?

1. **Edge Caching**: Responses served from Cloudflare's global network (faster)
2. **Cost Reduction**: Cached responses don't hit backend (saves LLM API costs)
3. **Rate Limiting**: Protects backend from abuse at the edge
4. **CORS Validation**: Additional security layer at Cloudflare
5. **Analytics**: Cloudflare provides request analytics
6. **DDoS Protection**: Cloudflare's DDoS protection for free

### Direct Access Still Possible

Backend is still accessible directly at `https://hedtools.ucsd.edu/hed-bot` but:
- Requires API key (X-API-Key header)
- No caching
- No edge rate limiting
- Subject to Nginx rate limiting only

## Production Considerations

1. **Backend Uptime**: Ensure Docker container is set for auto-restart
2. **API Key Rotation**: Rotate backend API key quarterly
3. **Cache Invalidation**: Deploy new worker to clear cache if needed
4. **Monitoring**: Set up Cloudflare alerts for error rates
5. **Costs**: Worker requests are free up to 100k/day, KV reads/writes have limits

## Alternative: Direct Connection

If you don't need caching or edge features, frontend can connect directly to:
```
https://hedtools.ucsd.edu/hed-bot
```

But you'll need to:
1. Add your frontend origin to backend CORS (EXTRA_CORS_ORIGINS)
2. Handle API key in frontend (less secure, visible in browser)
3. No caching benefits

Worker proxy is recommended for production use.

---

**Last Updated**: December 2, 2025
