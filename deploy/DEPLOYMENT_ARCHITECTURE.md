# HED-BOT Deployment Architecture

This document explains the deployment architecture for `hedtools.ucsd.edu` with proper CORS and security configuration.

## Table of Contents

- [Overview](#overview)
- [Option 1: Direct Connection (Recommended)](#option-1-direct-connection-recommended)
- [Option 2: Through Cloudflare Worker Proxy](#option-2-through-cloudflare-worker-proxy)
- [Security Configuration](#security-configuration)
- [Setup Instructions](#setup-instructions)

---

## Overview

**Server**: `hedtools.ucsd.edu`
**Frontend**: `hed-bot.pages.dev` (Cloudflare Pages)
**Backend**: FastAPI + Docker on port 33427

**Goal**: Only allow requests from `hed-bot.pages.dev` frontend

---

## Option 1: Direct Connection (Recommended)

### Architecture

```
┌─────────────────────────────┐
│  hed-bot.pages.dev         │  ← Frontend (Cloudflare Pages)
│  (Static Site)             │
└──────────────┬──────────────┘
               │ HTTPS
               ▼
┌─────────────────────────────┐
│  hedtools.ucsd.edu         │  ← Your Server
│                            │
│  Nginx Reverse Proxy       │
│  /hed-bot → :33427         │
│  (CORS validation)         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Docker Container          │
│  hed-bot:latest            │
│  127.0.0.1:33427           │
│  (FastAPI + CORS)          │
└─────────────────────────────┘
```

### Why This is Recommended

✅ **Simple**: Direct connection, fewer moving parts
✅ **Fast**: No extra proxy hop
✅ **Secure**: CORS enforced at both Nginx and FastAPI layers
✅ **Maintainable**: Single deployment, easy to debug

### CORS Protection

**Two-Layer Defense:**

1. **Nginx Layer**: Only allows requests from `hed-bot.pages.dev`
2. **FastAPI Layer**: Validates Origin header programmatically

Even if someone bypasses Nginx (impossible without server access), FastAPI will reject the request.

---

## Option 2: Through Cloudflare Worker Proxy

### Architecture

```
┌─────────────────────────────┐
│  hed-bot.pages.dev         │  ← Frontend (Cloudflare Pages)
│  (Static Site)             │
└──────────────┬──────────────┘
               │ HTTPS
               ▼
┌─────────────────────────────┐
│  Cloudflare Worker         │  ← Proxy Worker
│  worker.hed-bot.pages.dev  │
│  (Origin validation)       │
└──────────────┬──────────────┘
               │ HTTPS
               ▼
┌─────────────────────────────┐
│  hedtools.ucsd.edu         │  ← Your Server
│                            │
│  Nginx Reverse Proxy       │
│  /hed-bot → :33427         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Docker Container          │
│  hed-bot:latest            │
│  127.0.0.1:33427           │
└─────────────────────────────┘
```

### When to Use This

Use Cloudflare Worker proxy if you need:

- ⚡ **Edge caching** for responses
- 🔐 **Extra security layer** at Cloudflare edge
- 📊 **Cloudflare Analytics** on API usage
- 🌍 **DDoS protection** at Cloudflare level
- 🔧 **Request transformation** before hitting your server

### Cloudflare Worker Code

Create a Worker at `worker.hed-bot.pages.dev`:

```javascript
// Cloudflare Worker for HED-BOT API Proxy
export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Only allow requests from hed-bot.pages.dev
    const origin = request.headers.get('Origin');
    const allowedOrigin = 'https://hed-bot.pages.dev';

    if (origin !== allowedOrigin) {
      return new Response('Forbidden: Invalid origin', {
        status: 403,
        headers: {
          'Content-Type': 'text/plain',
        },
      });
    }

    // Proxy to backend server
    const backendUrl = `https://hedtools.ucsd.edu/hed-bot${url.pathname}${url.search}`;

    // Forward the request
    const backendRequest = new Request(backendUrl, {
      method: request.method,
      headers: request.headers,
      body: request.body,
    });

    // Get response from backend
    const backendResponse = await fetch(backendRequest);

    // Clone response and add CORS headers
    const response = new Response(backendResponse.body, backendResponse);
    response.headers.set('Access-Control-Allow-Origin', allowedOrigin);
    response.headers.set('Access-Control-Allow-Credentials', 'true');
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');

    return response;
  },
};
```

### Worker Configuration

**wrangler.toml** (if using separate worker project):

```toml
name = "hed-bot-api-proxy"
main = "src/index.js"
compatibility_date = "2024-01-01"

[env.production]
route = { pattern = "worker.hed-bot.pages.dev/*", zone_name = "hed-bot.pages.dev" }
```

### Frontend Configuration

Update frontend to use Worker URL:

```javascript
// In hed-bot frontend
const API_URL = 'https://worker.hed-bot.pages.dev';

// All API calls go through Worker
fetch(`${API_URL}/annotate`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ description: '...' }),
});
```

---

## Security Configuration

### CORS Origins

**Production**: Only `https://hed-bot.pages.dev`
**Development**: `http://localhost:5173`, `http://localhost:3000`

### Environment Variables

Add to `.env` on server:

```bash
# No extra CORS origins by default
# EXTRA_CORS_ORIGINS=https://staging.hed-bot.pages.dev,https://dev.hed-bot.pages.dev
```

### Nginx Security Headers

Add these to your Nginx config:

```nginx
# Security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;

# CSP for API (more permissive than frontend)
add_header Content-Security-Policy "default-src 'self'" always;
```

### Rate Limiting (Optional)

Protect against abuse:

```nginx
# In http block
limit_req_zone $binary_remote_addr zone=hed_bot_limit:10m rate=60r/m;

# In location block
limit_req zone=hed_bot_limit burst=10 nodelay;
limit_req_status 429;
```

This allows:
- **60 requests per minute** per IP
- **Burst of 10** extra requests
- **429 status code** when exceeded

---

## Setup Instructions

### Option 1: Direct Connection (Recommended)

#### Step 1: Update CORS in FastAPI

Already done! The code now only allows `hed-bot.pages.dev`.

#### Step 2: Configure Nginx

```bash
# On hedtools.ucsd.edu server

# Copy Nginx config
sudo cp deploy/nginx-hedtools.conf /etc/nginx/conf.d/hed-bot.conf

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

#### Step 3: Deploy Docker Container

```bash
# Deploy production container
./deploy/deploy.sh prod

# Verify it's running
docker ps | grep hed-bot
curl http://localhost:33427/health
```

#### Step 4: Test CORS

```bash
# Should succeed (from allowed origin)
curl -H "Origin: https://hed-bot.pages.dev" \
     -I https://hedtools.ucsd.edu/hed-bot-api/health

# Should fail (from disallowed origin)
curl -H "Origin: https://evil.com" \
     -I https://hedtools.ucsd.edu/hed-bot-api/health
```

#### Step 5: Update Frontend

In your frontend (`hed-bot.pages.dev`), update API URL:

```javascript
const API_URL = 'https://hedtools.ucsd.edu/hed-bot';
```

---

### Option 2: Through Cloudflare Worker

#### Step 1: Create Cloudflare Worker

```bash
# In your frontend repository or separate worker project
cd workers/api-proxy

# Create worker
wrangler publish
```

#### Step 2: Configure Routes

In Cloudflare dashboard:
- Go to Workers & Pages
- Add route: `worker.hed-bot.pages.dev/*` → `hed-bot-api-proxy`

#### Step 3: Update Nginx

Allow requests from Worker (or keep same CORS config):

```nginx
# Allow requests from Worker origin
# Worker will validate the actual frontend origin
```

#### Step 4: Update Frontend

```javascript
const API_URL = 'https://worker.hed-bot.pages.dev';
```

---

## Comparison

| Feature | Direct | Worker Proxy |
|---------|--------|--------------|
| **Simplicity** | ✅ Simple | ⚠️ More complex |
| **Speed** | ✅ Fast | ⚠️ +1 hop |
| **Cost** | ✅ Free | ✅ Free (Workers tier) |
| **Caching** | ❌ No | ✅ Yes (edge) |
| **DDoS Protection** | ⚠️ Basic | ✅ Cloudflare level |
| **Analytics** | ⚠️ Nginx logs | ✅ Cloudflare Analytics |
| **Debugging** | ✅ Easy | ⚠️ Harder |
| **Maintenance** | ✅ Low | ⚠️ Higher |

---

## Recommendation

**Start with Option 1 (Direct Connection)**

Reasons:
1. ✅ Simpler to set up and maintain
2. ✅ Faster (no extra hop)
3. ✅ Easier to debug
4. ✅ CORS protection is sufficient
5. ✅ Can add Worker later if needed

**Add Option 2 (Worker) later if you need:**
- Edge caching for API responses
- More detailed analytics
- Additional DDoS protection layer

---

## Testing Checklist

- [ ] Docker container builds successfully
- [ ] Container runs on port 33427
- [ ] Nginx proxies /hed-bot correctly
- [ ] Health check accessible: `https://hedtools.ucsd.edu/hed-bot-api/health`
- [ ] CORS allows `hed-bot.pages.dev`
- [ ] CORS blocks other origins
- [ ] Frontend can make API requests
- [ ] Streaming endpoints work (if using)

---

## Troubleshooting

### CORS errors in browser console

**Error**: `Access to fetch at 'https://hedtools.ucsd.edu/hed-bot-api/annotate' from origin 'https://hed-bot.pages.dev' has been blocked by CORS policy`

**Solutions**:
1. Check Nginx is running and configured correctly
2. Verify FastAPI CORS settings
3. Check Origin header in request
4. Look at response headers in Network tab

### 502 Bad Gateway

**Cause**: Nginx can't reach Docker container

**Solutions**:
1. Check container is running: `docker ps | grep hed-bot`
2. Check container health: `curl http://localhost:33427/health`
3. Check Nginx error logs: `sudo tail -f /var/log/nginx/error.log`

### Rate limiting issues

**Error**: `429 Too Many Requests`

**Solutions**:
1. Increase rate limit in Nginx config
2. Check if legitimate traffic or abuse
3. Add IP whitelist for trusted sources

---

**Last Updated**: December 2, 2025
