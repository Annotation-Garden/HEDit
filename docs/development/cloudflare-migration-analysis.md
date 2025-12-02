# Cloudflare Workers Migration Analysis

**Issue**: #8 (v0.4.2)
**Date**: December 2, 2025
**Status**: Research Complete - Implementation Pending

## Executive Summary

**FINAL DECISION (Dec 2, 2025)**: Deploy to **existing HED-web server at the center** - **ZERO cost**.

After comprehensive analysis of hosting options including Cloudflare Workers, Render.com, Railway, Fly.io, and Modal, the best solution is to use the **same server currently hosting HED web tools**. This provides:
- ✅ **$0/month** - using existing infrastructure
- ✅ **100% cost savings** vs. GPU server or paid hosting
- ✅ **Proven deployment pattern** - same Docker approach as hed-web
- ✅ **Cloudflare proxy already configured**
- ✅ **Full LangGraph support** with Python runtime

**Implementation**: Adapt `hed-web/deploy/` scripts for hed-bot Docker deployment on port 33427 with URL prefix `/hed-bot`.

---

**Research Summary Below** (Alternative options analyzed - kept for reference)

## Current Architecture

### Technology Stack
- **Framework**: FastAPI (Python 3.12)
- **Agent Orchestration**: LangGraph (multi-agent workflow)
- **LLM Providers**:
  - Ollama (local, GPU-accelerated)
  - OpenRouter (cloud-based)
- **Validation**:
  - HED JavaScript validator (Node.js)
  - HED Python validator (hedtools)
- **Dependencies**:
  - langgraph>=0.2.0
  - langchain>=0.3.0
  - fastapi>=0.121.0
  - hedtools>=0.5.0
  - lxml>=5.3.0
  - beautifulsoup4>=4.12.3

### Key Endpoints
1. `POST /annotate` - Generate HED annotation from natural language
2. `POST /annotate-from-image` - Generate HED annotation from image
3. `POST /annotate/stream` - Generate annotation with streaming
4. `POST /validate` - Validate HED annotation string
5. `GET /health` - Health check

### Current Costs (GPU Server)
- **Infrastructure**: RTX 4090 GPU server (~$100-200/month cloud hosting)
- **LLM Serving**: Ollama (self-hosted, no API costs)
- **Storage**: Local file system (HED schemas ~500MB)

## Cloudflare Workers Platform Analysis

### Python Workers Capabilities (Beta)

**✅ Supported:**
- FastAPI framework (official support)
- Python 3.12 runtime via Pyodide (WebAssembly)
- Async/await operations
- HTTP clients (httpx, aiohttp)
- Pure Python packages from PyPI
- Packages compiled for Pyodide (lxml, beautifulsoup4)
- Workers AI integration (native LLM access)
- R2 storage (10GB free)

**❌ Not Supported:**
- **LangGraph** (not available in Pyodide)
- **LangChain Python** (limited/experimental support)
- Packages with C extensions not compiled for WebAssembly
- File system access (must use R2/KV)
- Long-running processes (CPU time limits)

### Pricing Structure

#### Workers Platform
- **Free Tier**: 100,000 requests/day
- **Paid Plan**: $5/month + $0.30 per million requests

#### Workers AI
- **Free Tier**: 10,000 Neurons/day (~10,000 LLM requests for small models)
- **Paid Tier**: $0.011 per 1,000 Neurons
- **Model Pricing**: $0.017-$0.660 per million tokens (varies by model)

#### R2 Storage (for HED schemas)
- **Free Tier**: 10GB storage + 10M reads/month
- **Paid Tier**: $0.015/GB-month (if exceeding 10GB)
- **Zero egress fees** (unlike S3)

### Cost Comparison

**Current (GPU Server):**
- Infrastructure: ~$150/month
- Total: **$150/month**

**Cloudflare Workers (Hybrid):**
- Workers Platform: $5/month
- Workers AI: $5-20/month (depending on usage)
- R2 Storage: $0/month (within free tier)
- OpenRouter (backup): Pay-per-use
- Total: **$10-25/month** (83-93% cost reduction)

## Critical Compatibility Issues

### 1. LangGraph Unavailability

**Problem**: LangGraph is not available in Pyodide/Workers Python runtime.

**Evidence**:
- Pyodide package list does not include langgraph
- LangChain has limited experimental support
- Pure Python wheels exist but dependencies are incompatible with WebAssembly

**Impact**: Cannot run multi-agent workflow orchestration on Workers.

**Workarounds**:
1. **Rewrite agents without LangGraph** (high effort)
2. **Use Workers AI directly** (lose multi-agent orchestration)
3. **Hybrid architecture** (recommended, see below)

### 2. HED Validation Tools

**hedtools Analysis**:
- Depends on lxml (✅ available in Pyodide)
- May have additional C dependencies (⚠️ needs testing)
- Python validator can likely run on Workers
- JavaScript validator requires separate service

**Solution**: Test hedtools compatibility; fall back to JavaScript validator via external service if needed.

### 3. File System Dependencies

**Current**: Loads HED schemas from local file system
**Workers**: No file system access

**Solution**: Store schemas in R2 (10GB free tier sufficient for ~500MB schemas)

## Recommended Migration Strategy

### Option A: Hybrid Architecture (Recommended)

**Architecture**:
```
┌─────────────────────┐
│  Cloudflare Workers │
│   (FastAPI Layer)   │
│                     │
│  - API endpoints    │
│  - Request routing  │
│  - Response format  │
└──────────┬──────────┘
           │
           ├─────────────────┐
           │                 │
           ▼                 ▼
    ┌─────────────┐   ┌─────────────┐
    │  Workers AI │   │ OpenRouter  │
    │   (Primary) │   │  (Backup)   │
    └─────────────┘   └─────────────┘
           │                 │
           └────────┬────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  LangGraph Agent │
         │  (External API)  │
         │                  │
         │  - Run on Modal  │
         │  - Or Railway    │
         │  - Or Fly.io     │
         └──────────────────┘
```

**Implementation**:
1. Deploy FastAPI app to Cloudflare Workers
2. Replace LangGraph orchestration with:
   - Direct LLM calls to Workers AI or OpenRouter
   - Sequential agent execution (no graph)
   - State management in Workers KV
3. Store HED schemas in R2
4. Use external validation service (or deploy validator to Workers if compatible)

**Pros**:
- ✅ 83-93% cost reduction
- ✅ Global edge distribution (low latency)
- ✅ Zero-config scaling
- ✅ No GPU management
- ✅ Keep FastAPI code structure

**Cons**:
- ⚠️ Lose LangGraph workflow visualization
- ⚠️ More complex state management
- ⚠️ Sequential execution may be slower
- ⚠️ Requires code refactoring

### Option B: Full Rewrite (Not Recommended)

Rewrite entire application without LangGraph/LangChain.

**Pros**:
- ✅ Native Workers optimization
- ✅ Simpler dependency tree

**Cons**:
- ❌ High development effort
- ❌ Lose agent orchestration patterns
- ❌ Harder to maintain
- ❌ Lose LangChain ecosystem integration

### Option C: Keep Current Architecture (Status Quo)

Continue with GPU server + Ollama.

**Pros**:
- ✅ No migration effort
- ✅ Full LangGraph support
- ✅ Local LLM control

**Cons**:
- ❌ High infrastructure costs
- ❌ Manual scaling
- ❌ GPU management overhead
- ❌ Single point of failure

## Phase 1: Proof of Concept (v0.4.2)

### Goals
1. Verify FastAPI compatibility on Workers
2. Test hedtools/validation tools
3. Implement R2 schema loading
4. Test Workers AI integration
5. Benchmark performance vs. current setup

### Implementation Steps

#### 1. Setup Wrangler CLI
```bash
# Install Wrangler
npm install -g wrangler

# Authenticate
wrangler login

# Create new Python Worker project
wrangler init hed-bot-worker --type python
```

#### 2. Minimal FastAPI Worker

Create a simplified version with:
- Single `/health` endpoint
- Basic `/validate` endpoint (hedtools test)
- Schema loading from R2
- Workers AI test call

#### 3. Test Validation Tools
```python
# Test if hedtools works on Workers
import asyncio
from hedtools import HedString, HedSchema

async def test_validation():
    schema = HedSchema("8.3.0")  # Load from R2
    hed_string = HedString("Sensory-event")
    issues = hed_string.validate(schema)
    return len(issues) == 0
```

#### 4. R2 Schema Storage
```bash
# Create R2 bucket
wrangler r2 bucket create hed-schemas

# Upload schemas
wrangler r2 object put hed-schemas/8.3.0/HED8.3.0.json \
  --file=../hed-schemas/8.3.0/HED8.3.0.json
```

#### 5. Workers AI Integration
```python
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI

# In Workers context
llm = CloudflareWorkersAI(
    account_id=env.CLOUDFLARE_ACCOUNT_ID,
    api_token=env.CLOUDFLARE_API_TOKEN,
    model="@cf/meta/llama-3.1-8b-instruct"
)

response = await llm.ainvoke("Generate HED tags for: person sees red circle")
```

### Success Criteria
- ✅ FastAPI runs on Workers
- ✅ hedtools validation works
- ✅ Schemas load from R2 (<500ms)
- ✅ Workers AI responds (<2s)
- ✅ Total request time <5s

### Failure Points
- ❌ hedtools won't install (use JS validator instead)
- ❌ Workers AI too slow (use OpenRouter)
- ❌ Python Workers unstable (wait for GA)

## Phase 2: Full Migration (v0.5.0+)

If PoC succeeds:

1. **Rewrite Agent Orchestration**
   - Replace LangGraph with sequential execution
   - Implement retry logic manually
   - Use Workers KV for state persistence

2. **Deployment Pipeline**
   - GitHub Actions → Wrangler deploy
   - Automated testing on Workers
   - Staged rollout (canary deployment)

3. **Monitoring & Observability**
   - Cloudflare Analytics
   - Custom error logging to Workers KV
   - Performance metrics dashboard

4. **Documentation**
   - Update deployment guide
   - Add Workers-specific troubleshooting
   - Migration checklist for contributors

## Alternative: Keep Hybrid Long-Term

If full migration proves too complex:

**Architecture**:
- Cloudflare Workers: API layer + simple endpoints
- External service (Modal/Railway): LangGraph workflow
- Workers AI: Fast LLM inference
- OpenRouter: Complex reasoning models

**Benefits**:
- Best of both worlds
- Incremental migration
- Easier rollback
- Use right tool for each job

## Recommendations

### Short Term (v0.4.2)
1. ✅ **Start with Proof of Concept**
   - Deploy minimal FastAPI worker
   - Test critical dependencies
   - Benchmark performance
   - Estimated effort: 1-2 weeks

2. ✅ **Prepare Fallback Plan**
   - Keep GPU server running
   - Dual deployment during testing
   - Easy rollback strategy

### Long Term (v0.5.0+)
1. If PoC succeeds:
   - ✅ Migrate to hybrid architecture
   - ✅ Phase out GPU server
   - ✅ Redirect budget to development

2. If PoC fails:
   - ⚠️ Explore Modal AI or Railway for LangGraph
   - ⚠️ Keep GPU server for LLM serving
   - ⚠️ Use Workers for static content only

## Risk Assessment

### High Risk
- ❌ LangGraph incompatibility (confirmed)
- ⚠️ hedtools may not work (needs testing)
- ⚠️ Workers Python still in beta (stability concerns)

### Medium Risk
- ⚠️ Performance degradation (network latency)
- ⚠️ Workers AI model limitations (vs. local Ollama)
- ⚠️ Cold start times (Pyodide 6.4MB + packages)

### Low Risk
- ✅ Cost overruns (free tier generous, pay-per-use after)
- ✅ Vendor lock-in (FastAPI code portable)
- ✅ Scaling limits (100k requests/day free, virtually unlimited paid)

## Next Steps

1. **Immediate**: Present this analysis to stakeholders
2. **Week 1-2**: Implement PoC (Phase 1)
3. **Week 3**: Evaluate results, decide on full migration
4. **Week 4+**: Full migration or alternative solution

## Updated Recommendation (Dec 2, 2025)

### Key Insight: No GPU Needed

Since HED-BOT uses **OpenRouter exclusively** (not local Ollama), we don't need expensive GPU servers ($150/month). We just need a simple Python runtime for FastAPI + LangGraph.

### Recommended Platform: Render.com

**Why Render**:
- ✅ **Starter Plan**: $9/month (512MB RAM, 0.5 CPU, always-on)
- ✅ **Standard Plan**: $25/month (2GB RAM, 1 CPU, production-ready)
- ✅ Auto-deploy from GitHub
- ✅ Built-in monitoring and logs
- ✅ Free PostgreSQL database
- ✅ Zero configuration needed
- ✅ 70-90% cost reduction vs. GPU server

**Alternative Options**:
- **Railway**: $5/month + usage (~$15-30/month)
- **Fly.io**: Pay-as-you-go (~$5-15/month)
- ❌ **Modal**: NOT suitable (serverless, expensive for 24/7)

### Simplified Architecture

```
Cloudflare Pages (FREE)
        ↓
Render.com ($9-25/month)
  - FastAPI
  - LangGraph workflows
  - HED validation
        ↓
OpenRouter ($5-20/month usage)
  - LLM inference
  - Already using!
```

### Cost Breakdown

| Component | Current | New | Savings |
|-----------|---------|-----|---------|
| Infrastructure | $150 | $9-25 | 70-90% |
| LLM API | $0 (local) | $5-20 | N/A |
| **Total** | **$150** | **$14-45** | **70-90%** |

### Implementation Plan

**Week 1: Deploy to Render**
1. Create Render account
2. Connect GitHub repo
3. Configure environment variables
4. Deploy and test

**Week 2: Optimize**
1. Monitor performance
2. Optimize memory usage
3. Set up health checks
4. Configure alerts

**Week 3: Production**
1. Update frontend URLs
2. Configure custom domain
3. Retire GPU server
4. Migration complete!

### Success Criteria

- ✅ All LangGraph workflows function correctly
- ✅ Response times <5s
- ✅ Monthly cost <$30
- ✅ 99% uptime

### What We Learned from magland/qp

The [qp repository](https://github.com/magland/qp) demonstrates an even simpler approach:
- Uses Cloudflare Workers (TypeScript)
- Direct OpenRouter API calls (no LangGraph)
- D1 + R2 for storage
- **But**: They don't have multi-agent workflows

We need LangGraph for:
- Multi-agent orchestration
- Automatic retry loops
- Complex state management
- Workflow visualization

Therefore, we use Render for the Python backend while taking inspiration from qp's simple OpenRouter integration.

---

## Original Cloudflare Workers Analysis

*The sections below analyze Cloudflare Workers for Python applications. While technically feasible for simple FastAPI apps, it's not suitable for LangGraph-based workflows due to the Pyodide/WebAssembly limitations. Kept for reference.*

## References

- [Cloudflare Workers Python Docs](https://developers.cloudflare.com/workers/languages/python/)
- [Workers AI Pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/)
- [FastAPI on Workers](https://developers.cloudflare.com/workers/languages/python/packages/fastapi/)
- [Pyodide Packages](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)
- [Render Pricing](https://render.com/pricing)
- [Railway Pricing](https://railway.com/pricing)
- [Fly.io Pricing](https://fly.io/pricing)
- [Modal Pricing](https://modal.com/pricing)
- [magland/qp Repository](https://github.com/magland/qp)
- [Issue #8: Cloudflare Workers Migration](https://github.com/neuromechanist/hed-bot/issues/8)
