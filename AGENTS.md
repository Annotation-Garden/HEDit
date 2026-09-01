# HEDit Project Instructions

## Project Context
**Purpose:** Multi-agent system for converting natural language event descriptions
into valid Hierarchical Event Descriptors (HED) annotations using LangGraph.
Part of the Annotation Garden Initiative (AGI).
**Tech Stack:** Python 3.12+, LangGraph, FastAPI, HED JavaScript/Python validators,
Cloudflare Pages + Workers, Anthropic Claude via the Claude Platform on AWS
(Anthropic-operated Messages API, AWS Marketplace billing; NOT Bedrock).
Models: claude-haiku-4-5 (default) and claude-sonnet-5.

## Architecture Map
```
src/
├── agents/         # LangGraph agent implementations
├── validation/     # HED validation integration
├── api/            # FastAPI backend (main.py: endpoints, CORS, auth)
├── cli/            # Typer + Rich CLI
└── utils/          # Helper functions
frontend/           # Web interface (Cloudflare Pages; config.js picks backend)
workers/            # Cloudflare Worker proxies (index.js + wrangler.toml)
deploy/             # Deployment assets
tests/              # pytest + coverage
.context/           # Context files (architecture, deployment, HED semantics)
.rules/             # Rule files (git, python, testing, ci_cd, ...)
```

## Environment Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

uv run pytest                          # Run tests
uv run pytest -m "not integration"     # Skip integration tests
uv run pytest tests/test_hed_lsp.py -v # Single test file
```

## Deployment Architecture and Operations

### Request chain (production and dev)
```
hedit.pages.dev (frontend, Cloudflare Pages)
  └─> hedit-api.shirazi-10f.workers.dev (Cloudflare Worker proxy: Turnstile,
      rate limit, KV cache, CORS; forwards X-API-Key to backend)
        └─> api.annotation.garden/hedit (Cloudflare DNS -> Apache reverse
            proxy on SCCN VM, hedtools.ucsd.edu)
              └─> FastAPI container (port 38427)
```
Dev mirrors this: `develop.hedit.pages.dev` -> `hedit-dev-api.shirazi-10f.workers.dev`
-> `api.annotation.garden/hedit-dev` -> container port 38428.
`frontend/config.js` selects the Worker by hostname.

### Worker deployment (MANUAL - no CI)
The Workers in `workers/index.js` are NOT deployed by CI.
After any change to `workers/`, deploy both environments explicitly:
```bash
cd workers
bunx cfman wrangler --account neuromechanist deploy            # prod: hedit-api
bunx cfman wrangler --account neuromechanist deploy --env dev  # dev: hedit-dev-api
```
Secrets (`BACKEND_API_KEY`, `TURNSTILE_SECRET_KEY`) and KV bindings persist across deploys.

### Release pipeline (what a merge to main actually deploys, and when)
A merge to `main` does NOT immediately update the production backend.
The pipeline is:
1. CI on main bumps the version to the next alpha, tags it, and creates
   the GitHub release (resolving version conflicts in a develop->main PR:
   keep develop's `.dev` version; CI re-bumps after merge)
2. `docker-build.yml` builds and pushes the backend image to GHCR
   on every push to main (and develop)
3. The SCCN VM runs `deploy/auto-update.sh` from an hourly cron
   (minute 0), which pulls the new image and restarts the container --
   so the production API lags a merge by up to one hour.
   To deploy immediately, run `./deploy/auto-update.sh` on the VM
4. Cloudflare Pages deploys `hedit.pages.dev` from main and
   `develop.hedit.pages.dev` from develop automatically on push

Verify a backend deploy with
`curl https://api.annotation.garden/hedit/health` (reports the version);
the dev container follows the same pattern from develop pushes.

### CORS and request-header changes
The Worker maintains its own `Access-Control-Allow-Headers` list in `workers/index.js`,
separate from the FastAPI CORS middleware in `src/api/main.py`.
If the frontend starts sending a new request header (or the backend expects one),
BOTH lists must include it, and BOTH Workers must be redeployed.
A stale Worker list makes the browser preflight fail,
which surfaces as "Load failed" (Safari) or
"NetworkError when attempting to fetch resource" (Firefox).
Verify after deploying:
```bash
curl -si -X OPTIONS https://hedit-api.shirazi-10f.workers.dev/annotate/stream \
  -H 'Origin: https://hedit.pages.dev' \
  -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: content-type,x-anthropic-model,x-user-id' \
  | grep -i access-control
```

## Development Workflow
1. **Check context:** Review plan.md for current tasks and roadmap
2. **Branch:** Create feature branches from `develop`, merge back to `develop`
3. **Code:** Follow patterns (see `.rules/` for standards)
4. **Test:** `uv run pytest` with coverage; no mock tests
   (integration tests use real API calls with `ANTHROPIC_API_KEY` from `.env`)
5. **Commit:** Atomic, concise, no emojis, no AI attribution
6. **PR:** Target `develop` by default, not `main`
7. **Review:** Address ALL PR review findings; no technical debt carried forward

## Branching Strategy
- **main**: Production-ready code (stable releases)
- **develop**: Default target for PRs; active development branch (alpha releases)
- **Feature branches**: Create from develop, merge back to develop (dev releases)

## Versioning
- Use `scripts/bump_version.py` (never edit version manually)
- **Develop auto-bumps**: `auto-dev-bump.yml` increments `.devN` on every
  non-docs push to develop, so PRs to develop do NOT need a manual bump.
  A manual bump in a PR is harmless (the squash commit body carries the
  "Bump version to" line, which the workflow's loop guard detects and skips)
  but redundant; prefer letting CI bump
- **Version suffix rules by target branch:**
  - PRs to `develop`: `.dev` suffix (e.g., `0.6.8.dev0`)
  - PRs to `main`: `a` (alpha) suffix (e.g., `0.6.8a1`)
  - After merge to main for release: `b` (beta) or stable
- **Tags:** ONLY push after the PR is merged; never push tags from feature branches
- **Skip auto-release:** Add `[skip-release]` to commit messages on main
  for docs-only, context, or config changes
- Prerelease flow: `dev` (TestPyPI) -> `alpha`/`beta`/`stable` (PyPI)
- Example: `python scripts/bump_version.py patch --prerelease dev`

### Develop Branch Sync Rule
After syncing develop with main (post-release), bump patch and set to `.dev0`
(e.g., main at `0.6.7a2` -> develop becomes `0.6.8.dev0`).
Increment dev number for ongoing work.
Never use alpha versions on develop.
```
develop: 0.6.8.dev0 -> 0.6.8.dev1 -> 0.6.8.dev2 (TestPyPI)
    v (PR merge to main)
main: 0.6.8a1 -> 0.6.8a2 -> 0.6.8 (PyPI)
    v (sync back to develop)
develop: 0.6.9.dev0 (next cycle)
```

## [NEVER DO THIS]
- Never use mocks that replace business logic in tests
  (HTTP response fixtures like `respx` are acceptable for error/retry paths)
- Never use `pip`, `conda`, or `virtualenv`; use `uv`
- Never use `npm` or `npx`; use Bun (`bun`, `bunx`)
- Never commit secrets, `.env` files, or credentials
- Never edit version numbers manually; use `scripts/bump_version.py`
- Never push tags from feature branches
- Never change frontend/backend request headers without redeploying both Workers
- Never assume a merge to main is live in production; the backend container
  updates on the VM's hourly cron (see Release pipeline)

## [REFERENCE] Rules Directory
- `.rules/git.md` - Branching, commits, PRs, versioning
- `.rules/python.md` - UV, ruff, ty
- `.rules/testing.md` - No-mock policy, integration tests
- `.rules/ci_cd.md` - GitHub Actions standards
- `.rules/code_review.md` - PR review toolkit and checklist
- `.rules/documentation.md` - Documentation standards
- `.rules/self_improve.md` - Capturing learnings into rules

## Context Files
- `.context/agent-architecture.md` - Multi-agent system design
- `.context/api-and-deployment.md` - API endpoints, auth modes, hosting details
- `.context/hed-schemas.md` - HED schema structure and access
- `.context/hed-validation.md` - Validation tools and feedback
- `.context/hed-annotation-rules.md` - Core annotation semantics
- `plan.md` - Detailed roadmap and current tasks

## External Resources
- HED Schemas: `/Users/yahya/Documents/git/HED/hed-schemas`
- HED Validation: `/Users/yahya/Documents/git/HED/hed-javascript`
- HED Documentation: `/Users/yahya/Documents/git/HED/hed-resources`

## Deployment URLs
- **Production API**: https://api.annotation.garden/hedit
- **Development API**: https://api.annotation.garden/hedit-dev
- **Frontend**: https://hedit.pages.dev (dev: https://develop.hedit.pages.dev)
- **PyPI Package**: `hedit`

## Current Phase
See plan.md for the detailed roadmap and current tasks.
