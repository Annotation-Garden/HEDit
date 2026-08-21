# HEDit Project Overview

## Purpose
HEDit is a multi-agent system for converting natural language event descriptions into valid HED (Hierarchical Event Descriptors) annotation strings. Part of the Annotation Garden Initiative (AGI).

## Tech Stack
- **Language**: Python 3.12+
- **Package Manager**: uv
- **Agent Framework**: LangGraph
- **LLM Provider**: Anthropic Claude via the Claude Platform on AWS (claude-haiku-4-5 default, claude-sonnet-5 optional)
- **Validation**: HED JavaScript validator + HED Python tools (hedtools)
- **Backend**: FastAPI
- **CLI**: Typer + Rich
- **Frontend**: Cloudflare Pages
- **Workers**: Cloudflare Workers (API proxy)
- **API Hosting**: api.annotation.garden (SCCN VM via Apache reverse proxy)

## Current Version
- `0.7.0.dev1` on develop branch
- `0.6.8a2` on main branch

## Key Architecture
- Multi-agent workflow: annotate -> validate -> evaluate -> assess
- Iterative refinement with validation feedback loops
- Dual validation: JavaScript (preferred) + Python (fallback)
- State machine: `HedAnnotationState(TypedDict)` in `src/agents/state.py`
- Workflow orchestration: LangGraph StateGraph in `src/agents/workflow.py`

## Branching
- **main**: Production releases (alpha/beta/stable)
- **develop**: Active development (dev releases)
- Feature branches from develop, PRs target develop
