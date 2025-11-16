# HED-BOT Deployment Guide

This guide provides deployment options for HED-BOT.

## Deployment Options

### ⭐ Option 1: Cloudflare Pages + Workers (RECOMMENDED)
**Best for:** Using OpenRouter/Cerebras API (no local LLM)
- Fully serverless
- No backend infrastructure
- ~$2-5/month total cost
- 100,000 requests/day FREE
- See: `workers/README.md`

### Option 2: Cloudflare Pages + Tunnel
**Best for:** Using local LLM on your GPU machine
- Frontend on CDN
- Backend on local GPU
- $0/month (both free)
- See guide below

---

# Option 2: Pages + Tunnel Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  User Browser   │────────▶│ Cloudflare Pages │         │  Your GPU       │
│                 │         │   (Frontend CDN)  │         │   Machine       │
└─────────────────┘         └──────────────────┘         │                 │
                                      │                   │  ┌───────────┐  │
                                      │ API Requests      │  │  Backend  │  │
                                      └──────────────────▶│  │  FastAPI  │  │
                                        Cloudflare Tunnel │  │   + LLM   │  │
                                                           │  └───────────┘  │
                                                           └─────────────────┘
```

## Benefits

- **Fast deployment**: ~10 minutes to get live
- **Cost-effective**: Both Cloudflare services are free for basic use
- **Scalable**: Frontend on global CDN
- **Secure**: No port forwarding, tunnel handles encryption
- **GPU on your machine**: Keep expensive GPU workload local

---

## Part 1: Deploy Frontend to Cloudflare Pages

### Step 1: Push to GitHub

```bash
cd /home/yahya/git/hed-bot
git add .
git commit -m "Prepare for Cloudflare Pages deployment"
git push origin main
```

### Step 2: Create Cloudflare Pages Project

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Navigate to **Pages** → **Create a project**
3. **Connect to Git** → Select your GitHub account
4. Select the `hed-bot` repository
5. Configure build settings:
   - **Production branch**: `main`
   - **Build command**: (leave empty - it's a static site)
   - **Build output directory**: `frontend`
   - **Root directory**: `/`

6. Click **Save and Deploy**

### Step 3: Configure Backend URL

After deployment, you'll get a URL like: `https://hed-bot-abc.pages.dev`

Update `frontend/config.js` with your Cloudflare Tunnel URL (see Part 2):

```javascript
window.BACKEND_URL = 'https://your-tunnel-url.trycloudflare.com';
```

Then push the change:

```bash
git add frontend/config.js
git commit -m "Update backend URL for production"
git push origin main
```

Cloudflare Pages will auto-deploy on every push!

---

## Part 2: Expose Backend with Cloudflare Tunnel

### Step 1: Install cloudflared

On your GPU machine:

```bash
# Download cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb

# Install
sudo dpkg -i cloudflared-linux-amd64.deb

# Verify installation
cloudflared --version
```

### Step 2: Start Your Backend

Make sure your HED-BOT backend is running:

```bash
cd /home/yahya/git/hed-bot

# Start with Docker Compose
docker compose up -d hed-bot

# Or start directly
# conda activate hed-bot
# python -m uvicorn src.api.main:app --host 0.0.0.0 --port 38427
```

Verify it's running:
```bash
curl http://localhost:38427/health
```

### Step 3: Create Tunnel (Quick Start - Temporary URL)

For testing, create a quick tunnel (URL changes each time):

```bash
cloudflared tunnel --url http://localhost:38427
```

You'll see output like:
```
Your quick Tunnel has been created! Visit it at:
https://abc123.trycloudflare.com
```

**Copy this URL** and update `frontend/config.js`:

```javascript
window.BACKEND_URL = 'https://abc123.trycloudflare.com';
```

Then commit and push to update your Cloudflare Pages site.

### Step 4: Create Permanent Tunnel (Recommended for Production)

For a permanent, named tunnel:

1. **Login to Cloudflare**:
   ```bash
   cloudflared tunnel login
   ```

2. **Create a named tunnel**:
   ```bash
   cloudflared tunnel create hed-bot
   ```

3. **Create tunnel config** (`~/.cloudflared/config.yml`):
   ```yaml
   tunnel: hed-bot
   credentials-file: /home/yahya/.cloudflared/<TUNNEL-ID>.json

   ingress:
     - hostname: hed-bot.your-domain.com  # Your custom domain
       service: http://localhost:38427
     - service: http_status:404
   ```

4. **Add DNS record** (if using custom domain):
   ```bash
   cloudflared tunnel route dns hed-bot hed-bot.your-domain.com
   ```

5. **Run tunnel**:
   ```bash
   cloudflared tunnel run hed-bot
   ```

6. **Run as service** (auto-start on boot):
   ```bash
   sudo cloudflared service install
   sudo systemctl start cloudflared
   sudo systemctl enable cloudflared
   ```

---

## Part 3: Update CORS Settings

Update your FastAPI backend to allow requests from your Cloudflare Pages domain.

In `src/api/main.py`, update the CORS middleware:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://hed-bot-abc.pages.dev",  # Your Cloudflare Pages URL
        "https://your-custom-domain.com",  # Optional custom domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Restart your backend after making changes.

---

## Testing Your Deployment

1. **Frontend**: Visit your Cloudflare Pages URL
2. **Backend**: Check tunnel is running: `curl https://your-tunnel-url.trycloudflare.com/health`
3. **End-to-end**: Generate a test annotation through the web interface

---

## Monitoring

### Check Tunnel Status
```bash
cloudflared tunnel list
cloudflared tunnel info hed-bot
```

### View Tunnel Logs
```bash
sudo journalctl -u cloudflared -f
```

### Check Backend
```bash
docker compose logs -f hed-bot
```

---

## Costs

- **Cloudflare Pages**: Free (500 builds/month, unlimited requests)
- **Cloudflare Tunnel**: Free (unlimited bandwidth)
- **GPU Machine**: Your existing machine (no cloud GPU costs!)

---

## Alternative: Cloudflare Workers

For even more advanced setups, you could:
1. Deploy frontend to Pages (static)
2. Create a Cloudflare Worker as API proxy
3. Worker forwards requests to your tunnel
4. Adds caching, rate limiting, etc.

---

## Troubleshooting

### Frontend can't reach backend
- Check CORS settings in FastAPI
- Verify tunnel is running: `cloudflared tunnel list`
- Check backend URL in `config.js`

### Tunnel connection issues
- Check backend is running: `curl http://localhost:38427/health`
- Verify firewall isn't blocking cloudflared
- Check logs: `cloudflared tunnel info hed-bot`

### Cloudflare Pages build fails
- Ensure `frontend/` directory is in repo
- Check build settings (should be simple static site)
- Review build logs in Cloudflare dashboard

---

## Security Notes

1. **HTTPS Only**: Cloudflare Tunnel provides automatic HTTPS
2. **No Port Forwarding**: Tunnel handles all network security
3. **API Rate Limiting**: Consider adding rate limits to FastAPI
4. **Environment Variables**: Never commit sensitive data to git

---

## Next Steps

- Set up custom domain for Pages (optional)
- Configure permanent tunnel with custom domain
- Add monitoring/alerting
- Set up CI/CD for automatic deployments
- Consider adding authentication if needed
