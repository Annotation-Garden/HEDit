# HED-BOT Deployment Guide

This guide covers deploying HED-BOT on a GPU workstation with persistent URL access.

## Prerequisites

### Hardware
- NVIDIA GPU (tested on RTX 4090)
- CUDA 12.2+ installed
- Minimum 16GB RAM (32GB recommended for 10-15 concurrent users)
- Minimum 50GB disk space

### Software
- Docker with NVIDIA Container Toolkit
- Docker Compose
- Node.js 18+ (for JavaScript validator)
- Python 3.11+

## Quick Start with Docker

### 1. Clone and Setup

```bash
cd /path/to/hed-bot
cp .env.example .env
# Edit .env with your configuration
```

### 2. Build and Run

```bash
# Start all services with GPU support
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f hed-bot
```

### 3. Pull LLM Model

```bash
# Model is automatically pulled on first container start
# Or manually pull the default model:
docker exec -it hed-bot-ollama ollama pull gpt-oss:20b

# Alternative: Use a different model if needed
docker exec -it hed-bot-ollama ollama pull llama3.2
```

### 4. Access the Service

- API: http://localhost:38427
- Frontend: Open `frontend/index.html` in a browser
- API Docs: http://localhost:38427/docs

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
curl http://localhost:11434/api/tags
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
