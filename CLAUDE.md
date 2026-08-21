# HEDit Project

> **Note**: This project is being rebranded from `hed-bot` to `HEDit` and will migrate to the Annotation Garden organization. See plan.md for the v0.6.0 roadmap.

## Project Goal
Multi-agent system for converting natural language event descriptions into valid HED annotations using LangGraph. Part of the Annotation Garden Initiative (AGI).

## Tech Stack
- **Language**: Python 3.12+
- **Package Manager**: uv (fast Python package installer)
- **Agent Framework**: LangGraph
- **LLM Provider**: Anthropic Claude via the Claude Platform on AWS (Anthropic-operated Messages API, AWS Marketplace billing; NOT Bedrock). Models: claude-haiku-4-5 (default) and claude-sonnet-5.
- **Validation**: HED JavaScript validator + HED Python tools
- **Backend**: FastAPI
- **Frontend**: Cloudflare Pages
- **API Hosting**: api.annotation.garden (SCCN VM via Apache reverse proxy)

## Environment Setup
```bash
# Create virtual environment and install dev dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_hed_lsp.py -v
```

## Project Structure
```
hedit/                  # (currently hed-bot, renaming in v0.6.0)
├── src/
│   ├── agents/         # LangGraph agent implementations
│   ├── validation/     # HED validation integration
│   ├── utils/          # Helper functions
│   └── api/            # FastAPI backend
├── frontend/           # Web interface (Cloudflare Pages)
├── docs/               # Additional documentation
├── tests/              # Test suite with pytest + coverage
├── .context/           # Context files for Claude
├── .rules/             # Rule files for Claude
└── plan.md             # Detailed project planning (see this for tasks)
```

## Context Files
- `.context/hed-schemas.md`: HED schema structure and access
- `.context/hed-validation.md`: Validation tools and feedback
- `.context/hed-annotation-rules.md`: Core annotation semantics
- `.context/agent-architecture.md`: Multi-agent system design

## Branching Strategy
- **main**: Production-ready code (stable releases)
- **develop**: Default target for PRs; active development branch (alpha releases)
- **feature branches**: Create from develop, merge back to develop (dev releases)
- PRs should target `develop` by default, not `main`

## Versioning
- Use `scripts/bump_version.py` for version bumping (never edit version manually)
- **Version suffix rules by target branch**:
  - PRs to `develop`: Use `.dev` suffix (e.g., `0.6.8.dev0`)
  - PRs to `main`: Use `a` (alpha) suffix (e.g., `0.6.8a1`)
  - After merge to main for release: Use `b` (beta) or stable
- **Tag rules**:
  - Tags should ONLY be pushed after the PR is merged
  - Never push tags from feature branches
- **Skip auto-release**: Add `[skip-release]` to commit message to prevent CI version bumps on main (use for docs-only, context, or config changes)
- Prerelease labels:
  - `dev`: Feature branches being merged to develop (goes to TestPyPI)
  - `alpha`: develop branch merged to main (goes to PyPI as pre-release)
  - `beta`: Release candidates on main
  - `stable`: Production releases on main
- Example: `python scripts/bump_version.py patch --prerelease dev`

### Develop Branch Sync Rule
After syncing develop with main (post-release), always bump patch and set to `.dev0`:
- Example: main at `0.6.7a2` → develop becomes `0.6.8.dev0`
- For ongoing work on develop, increment dev number: `.dev0` → `.dev1` → `.dev2`
- Never use alpha versions on develop; keep alpha exclusive to main

**Version flow:**
```
develop: 0.6.8.dev0 → 0.6.8.dev1 → 0.6.8.dev2 (TestPyPI)
    ↓ (PR merge to main)
main: 0.6.8a1 → 0.6.8a2 → 0.6.8 (PyPI)
    ↓ (sync back to develop)
develop: 0.6.9.dev0 (next cycle)
```

This keeps clear separation: TestPyPI = dev builds, PyPI = alpha/beta/stable releases.

## Development Guidelines
1. **Atomic commits**: Make small, focused commits for each logical change
2. **Testing**: Use `uv run pytest` with coverage; avoid mock tests; integration tests use real API calls with the ANTHROPIC_API_KEY credentials from .env
3. **Linting**: Ruff pre-commit hooks with --fix and --unsafe-fixes
4. **Documentation**: Keep plan.md updated with progress
5. **Integration tests**: Run with `uv run pytest -m integration`; skip with `uv run pytest -m "not integration"`

## External Resources
- HED Schemas: `/Users/yahya/Documents/git/HED/hed-schemas`
- HED Validation: `/Users/yahya/Documents/git/HED/hed-javascript`
- HED Documentation: `/Users/yahya/Documents/git/HED/hed-resources`

## Deployment URLs
- **Production API**: https://api.annotation.garden/hedit (planned)
- **Development API**: https://api.annotation.garden/hedit-dev (planned)
- **Frontend**: https://hedit.pages.dev (Cloudflare Pages, planned)
- **PyPI Package**: `hedit` (planned)

## Current Phase
See plan.md for detailed roadmap and current tasks.
- **v0.6.0**: HEDit rebrand + api.annotation.garden migration
- **v0.6.1**: CLI implementation
- Use `uv` for all Python development