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
- Production: `annotation.garden` (pages calling the API directly; the frontend goes through the edge Worker, which does not forward `Origin`)
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

### Frontend (hedit-web Worker)
- URL: `https://annotation.garden/hedit/` (dev: `https://annotation.garden/hedit-dev/`)
- Static HTML/CSS/JS from `frontend/`, served by a Worker with static assets
  (`workers/web/`, routes `annotation.garden/hedit/*` and the bare `/hedit`)
- Calls the edge Worker at `edge.annotation.garden/hedit` (dev: `/hedit-dev`)
- SSE streaming support
- Legacy: the `hedit` Pages project (`hedit.pages.dev`, `develop.hedit.pages.dev`)
  still deploys from git until it is retired; its `config.js` redirects visitors
  to `annotation.garden/hedit/` and `/hedit-dev/`

### Cloudflare Workers (deployed by CI)
- Edge proxy: `workers/index.js`, config `workers/wrangler.toml`.
  Prod `hedit-api` at `edge.annotation.garden/hedit` -> `api.annotation.garden/hedit`;
  dev `hedit-dev-api` at `edge.annotation.garden/hedit-dev` -> `api.annotation.garden/hedit-dev`
- Frontend: `workers/web/` (`hedit-web`, dev `hedit-web-dev`), see above
- Secrets: Infisical, `annot-garden` project, `/hedit/edge` (prod and dev)
- Deploy: `.github/workflows/deploy-workers.yml` on a push to `develop` (dev)
  or `main` (prod) that touches `workers/` or `frontend/`, edge Worker first.
  Manual fallback: `cd workers && bunx cfman wrangler --account neuromechanist deploy`
  and `... deploy --env dev` (same in `workers/web`)
- The Worker keeps its own CORS `Access-Control-Allow-Headers` list;
  a change to it reaches both environments only once it is on `develop` and `main`,
  so release it before a client starts sending a new header
  ("Load failed" in Safari, "NetworkError" in Firefox)
- See AGENTS.md "Deployment Architecture and Operations" for the full chain
  and the preflight verification curl

## CLI (`src/cli/`)
- Built with Typer + Rich
- Commands: `annotate`, `annotate-image`, `validate`, `health`, `init`, `config`
- Config: `~/.config/hedit/config.yaml` + `credentials.yaml`
- Supports both local and remote API modes
