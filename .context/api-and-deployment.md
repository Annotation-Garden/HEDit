# API and Deployment Architecture

## FastAPI Backend (`src/api/main.py`)

### Endpoints
- `POST /annotate`: Text-to-HED annotation
- `POST /annotate/stream`: Streaming annotation via Server-Sent Events (SSE)
- `POST /annotate-from-image`: Image-to-HED annotation
- `POST /annotate-from-image/stream`: Streaming image annotation
- `POST /validate`: Standalone HED validation
- `POST /feedback`: User feedback submission
- `GET /health`: Service health check
- `GET /version`: Version information
- `GET /metrics`: Token use, cost, and prompt-cache savings since startup
  (server API key required; BYOK callers get 403)

### Authentication Modes
1. **Server mode**: `X-API-Key` header (server's Anthropic credentials are used)
2. **BYOK mode**: `X-Anthropic-Key` header (user's own Anthropic key, `sk-ant-...`,
   routed to api.anthropic.com)
3. **Public endpoints**: `/feedback`, `/health`, `/version`

### Model Override Headers
- `X-Anthropic-Model`: Override annotation model
- `X-Anthropic-Eval-Model`: Override evaluation model
- `X-Anthropic-Vision-Model`: Override vision model
- `X-Anthropic-Temperature`: Override temperature

The legacy `X-OpenRouter-*` spellings of these names (plus `X-OpenRouter-Key`) are
still accepted as transport; provider-routing headers are ignored.

### CORS
- Production: `hedit.pages.dev`, `annotation.garden`
- Development: Cloudflare Workers, localhost
- Custom: `EXTRA_CORS_ORIGINS` environment variable

## Deployment

### Docker
- Image: `hedit-api`
- Ports: 38427 (production), 38428 (development)
- Includes: Python, Node.js, HED schemas, JavaScript validator

### API Hosting (api.annotation.garden)
- Cloudflare DNS: CNAME `api` -> `hedtools.ucsd.edu`
- Cloudflare SSL: Origin Certificate for SCCN VM
- Apache reverse proxy: `/hedit` (prod), `/hedit-dev` (dev)

### Frontend (Cloudflare Pages)
- URL: `https://hedit.pages.dev`
- Static HTML/CSS/JS
- Cloudflare Workers proxy for API routing
- SSE streaming support

### Cloudflare Workers (manual deploy, no CI)
- Code: `workers/index.js`, config: `workers/wrangler.toml`
- Prod: `hedit-api` -> `api.annotation.garden/hedit`;
  dev: `hedit-dev-api` -> `api.annotation.garden/hedit-dev`
- Deploy (Cloudflare account `neuromechanist`):
  `cd workers && bunx cfman wrangler --account neuromechanist deploy`
  and `... deploy --env dev`
- The Worker keeps its own CORS `Access-Control-Allow-Headers` list;
  redeploy BOTH Workers whenever request headers change,
  otherwise browser preflights fail
  ("Load failed" in Safari, "NetworkError" in Firefox)
- See AGENTS.md "Deployment Architecture and Operations" for the full chain
  and the preflight verification curl

## CLI (`src/cli/`)
- Built with Typer + Rich
- Commands: `annotate`, `annotate-image`, `validate`, `health`, `init`, `config`
- Config: `~/.config/hedit/config.yaml` + `credentials.yaml`
- Supports both local and remote API modes
