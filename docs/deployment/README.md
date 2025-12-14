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
# HED-BOT Deployment Guide

This guide covers deploying HED-BOT on a GPU workstation with persistent URL access.

## Prerequisites

### Hardware
- NVIDIA GPU (tested on RTX 4090)
- CUDA 12.2+ installed
- Minimum 16GB RAM (32GB recommended for 10-15 concurrent users)
- Minimum 50GB disk space (includes model + HED resources)

### Software
- Docker with NVIDIA Container Toolkit
- Docker Compose

**Note**: Python, Node.js, HED schemas, and HED JavaScript validator are all included in the Docker image. No external dependencies needed!

## Quick Start with Docker

### 1. Clone Repository

```bash
cd /path/to/hed-bot
```

### 2. Build and Run (Self-Contained)

```bash
# Build and start all services
# This will:
# - Build Docker image with HED schemas and validator
# - Start Ollama and HED-BOT containers
# - Automatically pull gpt-oss:20b model on first start
docker-compose up -d

# Monitor first start (model download takes ~10-20 min)
docker-compose logs -f

# Check status
docker-compose ps
```

**What's Included in the Image:**
- Python 3.11 + all dependencies
- HED schemas (latest from GitHub)
- HED JavaScript validator (built)
- All self-contained, no external paths needed!

### 3. Model Auto-Pull

The `gpt-oss:20b` model is automatically pulled on first start. No manual intervention needed!

### 4. Verify Deployment

```bash
# Check API health
curl http://localhost:38427/health

# Should return:
# {
#   "status": "healthy",
#   "version": "0.1.0",
#   "llm_available": true,
#   "validator_available": true
# }
```

### 5. Access the Service

- **API**: http://localhost:38427
- **Frontend**: Open `frontend/index.html` in a browser
- **API Docs**: http://localhost:38427/docs

## Manual Deployment (without Docker)

### 1. Setup Conda Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate hed-bot
```

### 2. Install HED JavaScript Validator

```bash
cd /Users/yahya/Documents/git/HED/hed-javascript
npm install
npm run build
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env:
# - Set LLM_BASE_URL to your Ollama server
# - Set HED_VALIDATOR_PATH to hed-javascript location
```

### 4. Start Ollama (if not running)

```bash
# On workstation with GPU
ollama serve

# In another terminal, pull model
ollama pull gpt-oss:20b
```

### 5. Start HED-BOT API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 38427 --workers 4
```

### 6. Serve Frontend

```bash
# Simple Python server
cd frontend
python -m http.server 3000
```

Or use any static file server (nginx, Cloudflare Pages, etc.)

## Production Deployment

### Expose via Persistent URL

#### Option 1: Cloudflare Tunnel (Recommended)

```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# Create tunnel
cloudflared tunnel create hed-bot

# Route traffic
cloudflared tunnel route dns hed-bot hed-bot.yourdomain.com

# Run tunnel
cloudflared tunnel --config ~/.cloudflared/config.yml run hed-bot
```

Example `~/.cloudflared/config.yml`:
```yaml
tunnel: <TUNNEL_ID>
credentials-file: /home/user/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: hed-bot.yourdomain.com
    service: http://localhost:38427
  - service: http_status:404
```

#### Option 2: Ngrok (Quick Testing)

```bash
ngrok http 8000
```

#### Option 3: Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name hed-bot.yourdomain.com;

    location / {
        proxy_pass http://localhost:38427;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Systemd Service (for auto-restart)

Create `/etc/systemd/system/hed-bot.service`:

```ini
[Unit]
Description=HED-BOT Annotation Service
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/hed-bot
Environment="PATH=/home/youruser/miniconda3/envs/hed-bot/bin"
ExecStart=/home/youruser/miniconda3/envs/hed-bot/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 38427 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable hed-bot
sudo systemctl start hed-bot
sudo systemctl status hed-bot
```

## Performance Tuning

### For 10-15 Concurrent Users

1. **Ollama Configuration**:
   - Set `OLLAMA_NUM_PARALLEL=15` for concurrent requests
   - Use `OLLAMA_MAX_LOADED_MODELS=1` to keep model in VRAM

2. **API Workers**:
   - Set `--workers 4` (or number of CPU cores)
   - Use `--timeout-keep-alive 300` for long-running requests

3. **GPU Memory**:
   - Monitor with `nvidia-smi`
   - Default model: `gpt-oss:20b` (optimized for RTX 4090)
   - Consider smaller models if memory constrained (e.g., `llama3.2:8b`)

4. **Caching**:
   - HED schemas are cached in memory
   - Consider Redis for session management

## Monitoring

### Health Checks

```bash
# Check API health
curl http://localhost:38427/health

# Check Ollama
curl http://localhost:11435/api/tags
```

### Logs

```bash
# Docker logs
docker-compose logs -f

# Systemd logs
journalctl -u hed-bot -f

# Ollama logs
docker logs hed-bot-ollama -f
```

### Metrics

Monitor:
- GPU usage: `nvidia-smi` or `watch -n 1 nvidia-smi`
- API latency: FastAPI built-in metrics
- Request queue: Custom monitoring endpoint

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
```

### Ollama Out of Memory

- Default model is `gpt-oss:20b` (optimized for RTX 4090)
- Use smaller model if needed: `llama3.2:8b`
- Reduce `OLLAMA_NUM_PARALLEL`
- Increase GPU memory limit in Docker

### Validation Timeouts

- Check Node.js installation
- Verify HED JavaScript validator path
- Consider using Python validator (set `USE_JS_VALIDATOR=false`)

## Security

### For Production

1. **API Authentication**: Add API key middleware
2. **CORS**: Configure `allow_origins` in `src/api/main.py`
3. **Rate Limiting**: Use nginx or FastAPI middleware
4. **HTTPS**: Use Cloudflare or Let's Encrypt
5. **Firewall**: Restrict access to necessary ports only

## Backup and Maintenance

### Regular Tasks

```bash
# Backup models
docker run --rm -v ollama_data:/data -v $(pwd):/backup alpine \
    tar czf /backup/ollama_backup.tar.gz /data

# Update HED schemas
cd /Users/yahya/Documents/git/HED/hed-schemas
git pull

# Update hed-bot
cd /path/to/hed-bot
git pull
docker-compose build
docker-compose up -d
```

## Scaling

### For More Users (15+)

1. **Load Balancer**: Use nginx or HAProxy
2. **Multiple Workers**: Deploy multiple API instances
3. **Separate LLM Server**: Dedicated vLLM server with Ray
4. **Database**: Add Redis for state management
5. **Queue**: Use Celery for async processing
