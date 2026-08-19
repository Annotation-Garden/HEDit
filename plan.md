# HEDit Project Plan

> **Note**: This project is being rebranded from `hed-bot` to `HEDit`. See v0.6.0 roadmap below.

## Project Overview

Multi-agent system for converting natural language event descriptions into valid HED annotations using LangGraph. Part of the Annotation Garden Initiative (AGI).

## Architecture

### Multi-Agent System (LangGraph)
1. **Annotation Agent**: Generates initial HED tags from natural language
2. **Validation Agent**: Checks HED compliance using validation tools
3. **Evaluation Agent**: Provides iterative refinement feedback
4. **Assessment Agent**: Compares generated tags against original descriptions

### Technology Stack
- **Agent Framework**: LangGraph
- **LLM Provider**: Anthropic Claude via the Claude Platform on AWS (migrated from OpenRouter on 2026-08-18); models claude-haiku-4-5 (default) and claude-sonnet-5
- **Validation**: HED JavaScript validator + HED Python tools
- **Backend**: FastAPI
- **Frontend**: Cloudflare Pages
- **API Hosting**: api.annotation.garden (SCCN VM)

### Key Resources
- HED Schemas: `/Users/yahya/Documents/git/HED/hed-schemas`
- HED Validation: `/Users/yahya/Documents/git/HED/hed-javascript`
- HED Documentation: `/Users/yahya/Documents/git/HED/hed-resources`

---

## Completed: v0.1.0 - v0.5.x

Core system fully implemented and deployed. Summary of completed work:

- **Project Setup**: Python environment, git, pre-commit hooks
- **Infrastructure**: LangGraph, FastAPI, HED validators (Python + JavaScript)
- **Agent Development**: All 4 agents with feedback loops and vocabulary constraints
- **Integration**: REST API with /annotate, /validate, /health endpoints
- **Testing**: Unit tests with pytest, 80%+ coverage
- **Deployment**: Docker Compose, OpenRouter integration

Development now follows issue-based tracking. See [GitHub Issues](https://github.com/neuromechanist/hed-bot/issues) for detailed tasks.

---

## Upcoming: v0.6.0 - HEDit Rebrand & Annotation Garden Migration

This release rebrands hed-bot to **HEDit** and migrates the API to the Annotation Garden Initiative infrastructure. Related issue: #42

### Overview

| Item | Current | Target |
|------|---------|--------|
| Project name | hed-bot | HEDit |
| Package name | hed-bot | hedit |
| Repository | neuromechanist/hed-bot | Annotation-Garden/hedit |
| API URL | (via hedtools.org proxy) | api.annotation.garden/hedit |
| Dev API URL | - | api.annotation.garden/hedit-dev |
| PyPI | - | hedit |

### Phase 1: Infrastructure Setup

#### 1.1 Cloudflare DNS Configuration
Add CNAME record in Cloudflare dashboard for annotation.garden:
```
Type     Name    Target                 Proxy Status
CNAME    api     hedtools.ucsd.edu      Proxied (orange cloud)
```

#### 1.2 SSL: Cloudflare Origin Certificate
1. Cloudflare Dashboard → SSL/TLS → Origin Server → Create Certificate
2. Generate for `api.annotation.garden` (or `*.annotation.garden`)
3. Upload to SCCN VM:
```bash
sudo mkdir -p /etc/ssl/cloudflare
sudo nano /etc/ssl/cloudflare/api.annotation.garden.pem     # paste cert
sudo nano /etc/ssl/cloudflare/api.annotation.garden.key     # paste key
sudo chmod 600 /etc/ssl/cloudflare/api.annotation.garden.key
```
4. Set Cloudflare SSL mode to "Full (strict)"

#### 1.3 Apache Configuration on SCCN VM
```bash
sudo nano /etc/apache2/sites-available/api.annotation.garden.conf
```

```apache
<VirtualHost *:443>
    ServerName api.annotation.garden

    SSLEngine on
    SSLCertificateFile /etc/ssl/cloudflare/api.annotation.garden.pem
    SSLCertificateKeyFile /etc/ssl/cloudflare/api.annotation.garden.key

    # Production HEDit
    ProxyPass /hedit http://localhost:38427
    ProxyPassReverse /hedit http://localhost:38427

    # Development HEDit
    ProxyPass /hedit-dev http://localhost:38428
    ProxyPassReverse /hedit-dev http://localhost:38428

    # CORS headers
    Header always set Access-Control-Allow-Origin "*"
    Header always set Access-Control-Allow-Methods "GET, POST, OPTIONS"
    Header always set Access-Control-Allow-Headers "Content-Type, Authorization, X-OpenRouter-Key"
</VirtualHost>

<VirtualHost *:80>
    ServerName api.annotation.garden
    Redirect permanent / https://api.annotation.garden/
</VirtualHost>
```

```bash
sudo a2enmod proxy proxy_http ssl headers
sudo a2ensite api.annotation.garden.conf
sudo apache2ctl configtest
sudo systemctl reload apache2
```

### Phase 2: Project Rebrand

#### 2.1 Code Changes
- [ ] Rename package in `pyproject.toml` (hed-bot → hedit)
- [ ] Update Docker container names (hed-bot-api → hedit-api)
- [ ] Update all internal references and imports
- [ ] Update frontend branding
- [ ] Update CORS settings for api.annotation.garden
- [ ] Implement `X-OpenRouter-Key` header support for BYOK (bring your own key)

#### 2.2 Authentication Strategy
| Client | API Key | Behavior |
|--------|---------|----------|
| Web frontend | Embedded default key | Rate-limited, demo/casual use |
| CLI (no key) | User provides OpenRouter key | Via `X-OpenRouter-Key` header |

### Phase 3: Repository Migration

#### 3.1 GitHub Transfer
- [ ] Create `hedit` repo in Annotation-Garden org
- [ ] Transfer issues (open and closed) using GitHub CLI or API
- [ ] Archive neuromechanist/hed-bot with redirect notice
- [ ] Update all documentation links

#### 3.2 Issue Migration Script
```bash
# Export issues from source repo
gh issue list --repo neuromechanist/hed-bot --state all --json number,title,body,labels,state > issues.json

# Create issues in target repo (requires scripting)
# Note: PRs cannot be transferred, only issues
```

### Phase 4: Documentation Updates
- [ ] Update README.md with new name and URLs
- [ ] Update CLAUDE.md
- [ ] Add HEDit to annotation.garden website
- [ ] Cross-link from hedtags.org and hed-resources

### Infrastructure Checklist
```
[ ] Cloudflare: Add CNAME record (api → hedtools.ucsd.edu)
[ ] Cloudflare: Generate Origin Certificate for api.annotation.garden
[ ] Cloudflare: Set SSL mode to "Full (strict)"
[ ] SCCN VM: Upload Origin Certificate files
[ ] SCCN VM: Create Apache vhost config
[ ] SCCN VM: Enable Apache modules and site
[ ] SCCN VM: Run HEDit on ports 38427 (prod) and 38428 (dev)
[ ] Test: curl https://api.annotation.garden/hedit/health
```

---

## In Progress: v0.6.1 - Command-Line Interface (CLI)

Related issue: #9

### Overview
The `hedit` CLI provides command-line access to HEDit annotation capabilities using the hosted API at api.annotation.garden/hedit with BYOK (Bring Your Own Key) support.

### Implementation Status
- [x] CLI structure with Typer + Rich
- [x] Config management (API key storage, model settings)
- [x] Core commands: `annotate`, `annotate-image`, `validate`, `health`
- [x] PyPI publishing workflow (GitHub Actions)
- [x] Unit tests (53 tests passing)
- [ ] Integration tests with real API
- [ ] Documentation update

### Tech Stack
- **CLI Framework**: Typer 0.20.0+ (same author as FastAPI)
- **Terminal Output**: Rich 14.0.0+ (beautiful formatting)
- **Config Storage**: YAML in `~/.config/hedit/`
- **HTTP Client**: httpx 0.28.0+

### Features
- [x] `hedit init --api-key KEY` - Configure API key and preferences
- [x] `hedit annotate "description"` - Convert NL to HED
- [x] `hedit annotate-image image.png` - Annotate from image
- [x] `hedit validate "HED-string"` - Validate HED string
- [x] `hedit health` - Check API status
- [x] `hedit config show/set/path` - Manage configuration
- [x] `--api-key` flag (or `OPENROUTER_API_KEY` env var)
- [x] `--api-url` flag for custom endpoint
- [x] `-o json/text` output formats
- [x] `--schema` to specify HED version

### Installation
```bash
pip install hedit
```

### Usage
```bash
# Initialize with your OpenRouter API key
hedit init --api-key sk-or-xxx

# Generate annotation
hedit annotate "participant pressed the left button"

# From image
hedit annotate-image stimulus.png

# Validate HED string
hedit validate "Sensory-event, Visual-presentation"

# JSON output for scripting
hedit annotate "red circle appears" -o json > result.json

# Check API health
hedit health
```

### Configuration
Config stored in `~/.config/hedit/`:
- `config.yaml` - Settings (API URL, model, schema version)
- `credentials.yaml` - API key (chmod 600)

---

## Planned: v0.6.4 - User ID & Telemetry System

Related issues: #65 (User ID), #28 (Telemetry), #64 (Model exploration)

### Overview

Two independent features with different purposes:

**1. User ID (Issue #65)**: Cache optimization for BYOK users
- **Purpose**: Reduce API costs via OpenRouter sticky caching (>50% savings)
- **Privacy**: Anonymous, ephemeral, NOT stored or logged
- **Benefit**: Lower costs for users with their own API keys

**2. Telemetry (Issue #28)**: Training data collection
- **Purpose**: Collect annotation data for fine-tuning smaller models
- **Default**: ON (with clear disclosure)
- **Opt-out**: Available via config or `--no-telemetry` flag

**Critical Distinction**: User ID is for caching only and is NOT part of telemetry. We don't save user IDs.

### Phase 1: User ID Implementation (Issue #65)

#### 1.1 CLI: Machine-Based Stable ID
```python
# ~/.config/hedit/machine_id
# Generated once, persists across updates
def get_user_id():
    config_dir = Path.home() / ".config" / "hedit"
    id_file = config_dir / "machine_id"

    if id_file.exists():
        return id_file.read_text().strip()

    machine_id = uuid.uuid4().hex[:16]
    id_file.write_text(machine_id)
    return machine_id
```

Tasks:
- [ ] Implement machine ID generation in CLI config module
- [ ] Pass `user` parameter to OpenRouter API in all requests
- [ ] Add to `hedit init` workflow
- [ ] Update config documentation

#### 1.2 API: User ID Derivation

**BYOK (Bring Your Own Key)**:
- Derive from API key hash: `sha256(api_key)[:16]`
- Each key gets its own cache lane
- No PII stored

**Hosted Frontend**:
- Session-based ID (cookie) for anonymous users
- Fallback: `sha256(CF-Connecting-IP + date)[:16]`

Tasks:
- [ ] Add user ID derivation in FastAPI middleware
- [ ] Forward user ID to OpenRouter
- [ ] Implement session management for frontend

### Phase 2: Telemetry Infrastructure (Issue #28)

**Key Changes from Original Issue**:
- Changed from opt-in to **on by default**
- Clear disclosure on first run
- Easy opt-out mechanism

#### 2.1 User Disclosure

**CLI First Run**:
```
╭─ Welcome to HEDit! ─────────────────────────────────────╮
│                                                          │
│ HEDit collects anonymous usage data to improve the      │
│ annotation service.                                     │
│                                                          │
│ We collect (only from premium models):                  │
│   • Input descriptions and generated annotations        │
│   • Validation results and iteration counts             │
│   • Model performance metrics                           │
│                                                          │
│ Smart filtering reduces data collection:                │
│   • Skip open-source models (e.g., GPT-OSS)             │
│   • Skip duplicate inputs                               │
│                                                          │
│ We DO NOT collect:                                      │
│   • Personal information or API keys                    │
│   • User IDs (used only for caching, not logged)        │
│                                                          │
│ To disable: hedit config set telemetry.enabled false    │
│                                                          │
╰──────────────────────────────────────────────────────────╯
```

**When to Show**:
- On `hedit init` (first-time setup)
- First time after installation (check `~/.config/hedit/.first_run`)
- Can be reviewed with `hedit config show`

**Website**:
- Similar notice in footer or FAQ
- Link to privacy policy

#### 2.2 Data Collection

**Smart Filtering** (reduces storage by ~70-80%):

1. **Model Blacklist**: Only collect from premium models
   - **Rationale**: Use expensive model outputs to fine-tune cheaper models
   - **Default blacklist**: `["openai/gpt-oss-120b"]` (current default)
   - **Collect from**: Claude Haiku 4.5, GPT-4o, other premium models
   - **Configurable**: `telemetry.model_blacklist` in config

2. **Input Deduplication**: Skip repeated descriptions
   - **Method**: Hash-based KV keys (`telemetry:{sha256(input)[:16]}:{event_id}`)
   - **Process**:
     - Hash input description
     - Check if hash exists in KV (1 read)
     - If new: Write telemetry (1 write)
     - If duplicate: Skip write (saves write operation)
   - **Impact**: ~50% reduction in writes for typical usage

**Expected Savings**:
- 100 annotations/day, 50% duplicates, 80% from blacklisted models
- Actual writes: 100 → 10/day (90% reduction!)
- Still within free tier: 10 writes/day << 1,000/day limit

**What to collect** (when filters pass):
- Session ID (ephemeral, for grouping related requests)
- Input: original description (hashed for dedup)
- Model: provider, name, parameters
- Output: generated HED string
- Validation: errors, warnings, quality score
- Metadata: timestamp, schema version, iteration count
- Performance: latency, token count, cost

**What NOT to collect**:
- User IDs (ephemeral, used only for caching)
- Personal information (names, emails, IPs)
- API keys
- Data from blacklisted models
- Duplicate inputs

#### 2.3 Storage Strategy

**Development/Testing**:
- Local JSON files in `/tmp/hedit-telemetry/`
- CLI flag: `--no-telemetry` to disable

**Production Options**:

*Option 1: Cloudflare Workers KV* (Recommended)
- **Free Tier**: 1 GB storage, 100k reads/day, 1k writes/day
- **Cost Analysis**: For typical usage (~100 annotations/day):
  - Storage: ~1 MB/day → 365 MB/year (well within 1 GB free tier)
  - Writes: 100/day (well within 1k/day free tier)
  - Reads: Minimal (batch exports only)
- **Verdict**: **FREE for foreseeable future**
- **Benefits**: Fast, distributed, no server maintenance
- Reference: [Cloudflare KV Pricing](https://developers.cloudflare.com/kv/platform/pricing/)

*Option 2: SCCN Server Storage*
- Store in `/var/hedit-telemetry/` on api.annotation.garden
- Requires disk space management and backup
- More control but more maintenance

**Recommendation**: Start with Cloudflare KV (free tier is generous)

**KV Key Structure**:
```
Key: telemetry:{input_hash}:{event_id}
Example: telemetry:a3f5e2b8c1d4f6a9:550e8400-e29b-41d4-a716-446655440000
```

**Schema** (Note: No user_id field):
```json
{
  "event_id": "uuid",
  "input_hash": "sha256(description)[:16]",
  "session_id": "ephemeral_session_id",
  "timestamp": "2025-12-20T17:47:33Z",
  "input": {
    "description": "participant pressed left button",
    "schema_version": "8.3.0"
  },
  "output": {
    "hed_string": "Agent-action, Press, (Left, Button)",
    "iterations": 2,
    "validation_errors": []
  },
  "model": {
    "model": "anthropic/claude-haiku-4.5",
    "provider": null,
    "temperature": 0.1
  },
  "performance": {
    "latency_ms": 450,
    "input_tokens": 1200,
    "output_tokens": 85,
    "cost_usd": 0.0012
  },
  "source": "cli|api|web"
}
```

**Note on `provider` field**:
- OpenRouter routes requests automatically based on model name
- `provider` parameter is only needed for specific models (e.g., GPT-OSS, Qwen)
- For most models (Claude, GPT-4o, etc.): `provider = null`
- Telemetry captures the provider parameter if it was used

Tasks:
- [ ] Design telemetry schema with input_hash field
- [ ] Implement model blacklist filtering
  - [ ] Default blacklist: `["openai/gpt-oss-120b"]`
  - [ ] Config option: `telemetry.model_blacklist`
  - [ ] Skip telemetry if model in blacklist
- [ ] Implement input deduplication
  - [ ] Hash input: `sha256(description)[:16]`
  - [ ] Check KV for existing hash (1 read)
  - [ ] Skip write if duplicate found
  - [ ] Use hash-based KV keys: `telemetry:{hash}:{event_id}`
- [ ] Implement first-run disclosure notice (Rich panel)
- [ ] Add `.first_run` tracker in `~/.config/hedit/`
- [ ] Implement telemetry collector in API
- [ ] Implement telemetry in CLI
- [ ] Add Cloudflare KV storage backend
- [ ] Add local file fallback for development
- [ ] Add `--no-telemetry` flag
- [ ] Add `hedit config set telemetry.enabled false`
- [ ] Update privacy policy with filtering details

#### 2.4 Privacy & Compliance

**Disclosure Requirements**:
- [ ] Show notice on first `hedit init` or first command after install
- [ ] Add telemetry notice to README and website
- [ ] Update docs with data collection policy
- [ ] Add telemetry status to `hedit config show`
- [ ] Link to privacy policy from CLI help

**Opt-out Mechanisms**:
- [ ] `hedit config set telemetry.enabled false`
- [ ] `--no-telemetry` flag for one-off commands
- [ ] Environment variable: `HEDIT_TELEMETRY=0`

**Data Handling**:
- [ ] Document data retention policy (90 days rolling window)
- [ ] Implement automatic cleanup of old telemetry data
- [ ] Ensure user IDs are never stored (only used for OpenRouter caching)

### Phase 3: Data Analysis Pipeline

**Goal**: Enable model fine-tuning and performance comparison

Tasks:
- [ ] Export telemetry to training format (JSONL)
- [ ] Filter high-quality annotations (validation passed, <3 iterations)
- [ ] Generate statistics for model comparison
- [ ] Create dataset for fine-tuning GPT OSS or similar models

### Implementation Strategy

**Feature Branch Workflow**:

1. **Branch 1**: `feature/user-id-caching` → `develop`
   - Implement machine ID generation
   - Pass user ID to OpenRouter API
   - Version: `0.6.4-dev1`
   - PR to develop, test, merge

2. **Branch 2**: `feature/telemetry-infrastructure` → `develop`
   - Implement telemetry schema and collection
   - Add Cloudflare KV storage
   - Implement opt-out mechanisms
   - Version: `0.6.4-dev2`
   - PR to develop, test, merge

3. **Branch 3**: `feature/telemetry-disclosure` → `develop`
   - Add first-run disclosure notice
   - Update frontend with privacy notice
   - Update documentation (in `../documentation/`)
   - Version: `0.6.4-dev3`
   - PR to develop, test, merge

4. **Final Testing**: After all branches merged to develop
   - Integration testing with both features
   - Frontend disclosure verification
   - Documentation review
   - Tag and release: `v0.6.4-alpha` → `main`

**Documentation Updates**:
- Repository: `/Users/yahya/Documents/git/annot-garden/documentation`
- Files to update: Privacy policy, user guide, CLI reference

### Expected Outcomes

1. **Cost Optimization (User ID)**: >50% API cost reduction via prompt caching for BYOK users
2. **Training Data (Telemetry)**: High-quality annotation pairs for fine-tuning smaller models
3. **Model Selection**: Data-driven comparison for issue #64
4. **Performance Metrics**: Track quality improvements over time
5. **User Trust**: Transparent disclosure with easy opt-out builds confidence

---

## Backlog: v0.7.0+ - Model Exploration & Optimization

Related issue: #64

**Blocked by**: v0.6.2 telemetry (need data first)

### Tasks
- [ ] Analyze telemetry data for baseline performance metrics
- [ ] Evaluate alternative models: Claude Haiku 4.5, GPT-4o-mini, Llama 3 70B
- [ ] Compare: cost, quality, iteration count, latency
- [ ] Fine-tune GPT OSS on collected high-quality annotations
- [ ] A/B test new models with telemetry
- [ ] Update default model based on results

---

## Future Roadmap

See GitHub Issues for detailed tracking:
- #7: Events.tsv workflow with multi-agent system
- #12-#19: Extended validation and annotation agents
- #27: Health check improvements
