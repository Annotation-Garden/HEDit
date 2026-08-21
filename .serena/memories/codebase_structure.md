# Codebase Structure

```
hedit/
├── src/
│   ├── agents/                    # LangGraph multi-agent system
│   │   ├── workflow.py            # HedAnnotationWorkflow (orchestration)
│   │   ├── state.py               # HedAnnotationState (TypedDict)
│   │   ├── annotation_agent.py    # Generates HED tags from text
│   │   ├── validation_agent.py    # Validates HED compliance
│   │   ├── evaluation_agent.py    # Assesses faithfulness
│   │   ├── assessment_agent.py    # Final completeness check
│   │   ├── feedback_summarizer.py # Condenses feedback for LLM
│   │   ├── vision_agent.py        # Image-to-text via vision LLMs
│   │   └── feedback_triage_agent.py # User feedback processing
│   ├── validation/
│   │   ├── hed_validator.py       # Dual JS+Python validators
│   │   └── hed_lsp.py             # HED Language Server Protocol
│   ├── utils/
│   │   ├── hed_rules.py           # HED syntax/semantic rules (system prompts)
│   │   ├── hed_comprehensive_guide.py # Full LLM-optimized HED guide
│   │   ├── json_schema_loader.py  # JSON schema + vocabulary extraction
│   │   ├── schema_loader.py       # Legacy Python schema loader
│   │   ├── error_remediation.py   # Error augmentation for LLM feedback
│   │   ├── anthropic_llm.py       # Anthropic Claude integration (prompt caching)
│   │   ├── llm_usage.py           # Token, cost, and cache accounting
│   │   ├── litellm_llm.py         # Alternative LLM providers
│   │   ├── image_processing.py    # Base64 image encoding
│   │   └── github_client.py       # GitHub API for feedback issues
│   ├── api/
│   │   ├── main.py                # FastAPI app (endpoints)
│   │   ├── models.py              # Pydantic request/response models
│   │   └── security.py            # Authentication
│   ├── cli/
│   │   ├── main.py                # Typer CLI entry point
│   │   ├── config.py              # Config management (YAML)
│   │   ├── client.py              # API client
│   │   ├── executor.py            # Execution strategy
│   │   ├── local_executor.py      # Local workflow execution
│   │   ├── api_executor.py        # Remote API execution
│   │   ├── output.py              # Output formatting
│   │   └── commands/              # CLI subcommands
│   ├── telemetry/
│   │   ├── collector.py           # Data collection
│   │   ├── storage.py             # Local + Cloudflare KV
│   │   └── schema.py              # Telemetry data schema
│   ├── scripts/
│   │   └── process_feedback.py    # Feedback processing
│   └── version.py                 # Version info
├── tests/                         # pytest test suite (22 files)
├── docs/                          # Documentation
├── frontend/                      # Web UI (Cloudflare Pages)
├── workers/                       # Cloudflare Workers proxy
├── deploy/                        # Deployment configs
├── docker/                        # Docker configs
├── scripts/                       # Utility scripts (bump_version.py)
├── .context/                      # Context files for AI agents
├── .rules/                        # Development rules
├── pyproject.toml                 # Project config + dependencies
└── CLAUDE.md                      # AI agent instructions
```

## Key Entry Points
- **API**: `src/api/main.py` (FastAPI app)
- **CLI**: `src/cli/main.py` (Typer CLI, registered as `hedit` command)
- **Workflow**: `src/agents/workflow.py` (LangGraph state graph)
