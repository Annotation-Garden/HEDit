# HEDit Deployment Guide

This guide provides deployment options for HEDit.

## Deployment Options

### Option 1: Cloudflare Pages + Workers (RECOMMENDED)
**Best for:** Fully serverless use of the Claude Platform on AWS (no self-hosted backend)
- Fully serverless
- No backend infrastructure
- LLM usage billed through your AWS account
- 100,000 requests/day FREE
- See: `workers/README.md`

### Option 2: Cloudflare Pages + Tunnel
**Best for:** Hosting the backend on your own machine
- Frontend on CDN
- Backend on your own machine (LLM calls go to the Claude Platform on AWS)
- $0/month for Cloudflare (both services free)
- See guide below

---

# Option 2: Pages + Tunnel Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  User Browser   │────────▶│ Cloudflare Pages │         │  Your Server    │
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
- **Backend on your machine**: Full control over the API server; LLM inference runs on the Claude Platform on AWS

---

## Part 1: Deploy Frontend to Cloudflare Pages

### Step 1: Push to GitHub

```bash
cd /home/yahya/git/HEDit
git add .
git commit -m "Prepare for Cloudflare Pages deployment"
git push origin main
```

### Step 2: Create Cloudflare Pages Project

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Navigate to **Pages** → **Create a project**
3. **Connect to Git** → Select your GitHub account
4. Select the `HEDit` repository
5. Configure build settings:
   - **Production branch**: `main`
   - **Build command**: (leave empty - it's a static site)
   - **Build output directory**: `frontend`
   - **Root directory**: `/`

6. Click **Save and Deploy**

### Step 3: Configure Backend URL

After deployment, you'll get a URL like: `https://abc123.hedit.pages.dev`

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

On your backend machine:

```bash
# Download cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb

# Install
sudo dpkg -i cloudflared-linux-amd64.deb

# Verify installation
cloudflared --version
```

### Step 2: Start Your Backend

Make sure your HEDit backend is running:

```bash
cd /home/yahya/git/HEDit

# Start with Docker Compose
docker compose up -d hedit

# Or start directly
# conda activate hedit
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
   cloudflared tunnel create hedit
   ```

3. **Create tunnel config** (`~/.cloudflared/config.yml`):
   ```yaml
   tunnel: hedit
   credentials-file: /home/yahya/.cloudflared/<TUNNEL-ID>.json

   ingress:
     - hostname: hedit.your-domain.com  # Your custom domain
       service: http://localhost:38427
     - service: http_status:404
   ```

4. **Add DNS record** (if using custom domain):
   ```bash
   cloudflared tunnel route dns hedit hedit.your-domain.com
   ```

5. **Run tunnel**:
   ```bash
   cloudflared tunnel run hedit
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
        "https://abc123.hedit.pages.dev",  # Your Cloudflare Pages URL
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
cloudflared tunnel info hedit
```

### View Tunnel Logs
```bash
sudo journalctl -u cloudflared -f
```

### Check Backend
```bash
docker compose logs -f hedit
```

---

## Costs

- **Cloudflare Pages**: Free (500 builds/month, unlimited requests)
- **Cloudflare Tunnel**: Free (unlimited bandwidth)
- **Backend Machine**: Your existing machine (LLM usage billed through your AWS account)

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
- Check logs: `cloudflared tunnel info hedit`

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
# HEDit Deployment Guide

This guide covers deploying HEDit on your own workstation or server with persistent URL access.

## Prerequisites

### Hardware
- Minimum 8GB RAM (16GB recommended for 10-15 concurrent users)
- Minimum 10GB disk space (includes HED resources)
- No GPU required; LLM inference runs on the Claude Platform on AWS

### Software
- Docker
- Docker Compose
- Claude Platform on AWS credentials (see [claude-platform-aws.md](claude-platform-aws.md))

**Note**: Python, Node.js, HED schemas, and HED JavaScript validator are all included in the Docker image. No external dependencies needed!

## Quick Start with Docker

### 1. Clone Repository

```bash
cd /path/to/hedit
```

### 2. Build and Run (Self-Contained)

```bash
# Build and start all services
# This will:
# - Build Docker image with HED schemas and validator
# - Start the HEDit container
docker-compose up -d

# Monitor first start
docker-compose logs -f

# Check status
docker-compose ps
```

**What's Included in the Image:**
- Python 3.11 + all dependencies
- HED schemas (latest from GitHub)
- HED JavaScript validator (built)
- All self-contained, no external paths needed!

### 3. Configure LLM Credentials

Set the Claude Platform on AWS credentials in `.env` (all three are required):

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-api-key-here
ANTHROPIC_BASE_URL=https://aws-external-anthropic.us-east-2.api.aws
ANTHROPIC_WORKSPACE_ID=wrkspc_your_workspace_id
```

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
conda activate hedit
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
# - Set ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_WORKSPACE_ID
# - Set HED_VALIDATOR_PATH to hed-javascript location
```

### 4. Start HEDit API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 38427 --workers 4
```

### 5. Serve Frontend

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
cloudflared tunnel create hedit

# Route traffic
cloudflared tunnel route dns hedit hedit.yourdomain.com

# Run tunnel
cloudflared tunnel --config ~/.cloudflared/config.yml run hedit
```

Example `~/.cloudflared/config.yml`:
```yaml
tunnel: <TUNNEL_ID>
credentials-file: /home/user/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: hedit.yourdomain.com
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
    server_name hedit.yourdomain.com;

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

Create `/etc/systemd/system/hedit.service`:

```ini
[Unit]
Description=HEDit Annotation Service
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/hedit
Environment="PATH=/home/youruser/miniconda3/envs/hedit/bin"
ExecStart=/home/youruser/miniconda3/envs/hedit/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 38427 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable hedit
sudo systemctl start hedit
sudo systemctl status hedit
```

## Performance Tuning

### For 10-15 Concurrent Users

1. **API Workers**:
   - Set `--workers 4` (or number of CPU cores)
   - Use `--timeout-keep-alive 300` for long-running requests

2. **Model Selection**:
   - Default model: `claude-haiku-4-5` (fast, used for annotation, evaluation, and vision)
   - Optional: `claude-sonnet-5`, larger and 2.3x the cost, with no measured quality gain
     (see `docs/reasoning.md`)

3. **Caching**:
   - HED schemas are cached in memory
   - Consider Redis for session management

## Monitoring

### Health Checks

```bash
# Check API health
curl http://localhost:38427/health
```

### Logs

```bash
# Docker logs
docker-compose logs -f

# Systemd logs
journalctl -u hedit -f
```

### Metrics

Monitor:
- API latency: FastAPI built-in metrics
- Request queue: Custom monitoring endpoint

## Troubleshooting

### LLM Errors

- Verify `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, and `ANTHROPIC_WORKSPACE_ID` are set
  (the endpoint rejects requests without the workspace header)
- Check the key in the AWS Console under Claude Platform on AWS -> API keys

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
# Update HED schemas
cd /Users/yahya/Documents/git/HED/hed-schemas
git pull

# Update HEDit
cd /path/to/hedit
git pull
docker-compose build
docker-compose up -d
```

## Scaling

### For More Users (15+)

1. **Load Balancer**: Use nginx or HAProxy
2. **Multiple Workers**: Deploy multiple API instances
3. **Database**: Add Redis for state management
4. **Queue**: Use Celery for async processing
