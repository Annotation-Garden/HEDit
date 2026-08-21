# Claude Platform on AWS Integration

This document explains how to configure HEDit to use the Claude Platform on Amazon Web Services (AWS)
for cloud-based large language model (LLM) inference.

## What is the Claude Platform on AWS?

The Claude Platform on AWS is an Anthropic-operated Messages API billed through the AWS Marketplace.
It is explicitly NOT Amazon Bedrock:
requests go to an Anthropic-managed endpoint
and are authenticated with an Anthropic API key plus a workspace identifier.

- **Single provider**: All agents run on Anthropic Claude models
- **Simple billing**: Usage is billed through your AWS account
- **Consistent quality**: The same models power annotation, evaluation, and vision

## Offered Models

| Model | Role |
|-------|------|
| `claude-haiku-4-5` | Default for annotation, evaluation (judge), and vision |
| `claude-sonnet-5` | Optional, larger, 2.3x the cost (no measured quality gain, see `docs/reasoning.md`) |

No other models are offered (Opus is not available).
Legacy OpenRouter-style identifiers such as `anthropic/claude-haiku-4.5` are accepted as aliases,
but `qwen`, `mistral`, and `gpt-oss` model identifiers are rejected with HTTP 400.

## Getting Credentials

1. Sign in to the AWS Console
2. Navigate to **Claude Platform on AWS** -> **API keys**
3. Create an API key and note your workspace ID (`wrkspc_...`)

## Setup

### Environment Variables

All three credentials are required;
the endpoint rejects requests that do not carry the workspace header.

**Option A: Using .env file (Recommended)**

```bash
cp .env.example .env
```

Then edit `.env` and set:

```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-api-key-here
ANTHROPIC_BASE_URL=https://aws-external-anthropic.us-east-2.api.aws
ANTHROPIC_WORKSPACE_ID=wrkspc_your_workspace_id

# Model configuration (defaults shown)
ANNOTATION_MODEL=claude-haiku-4-5
EVALUATION_MODEL=claude-haiku-4-5
VISION_MODEL=claude-haiku-4-5
```

**Option B: Environment variables** (edit `~/.bashrc` or `~/.zshrc`):

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-api-key-here
export ANTHROPIC_BASE_URL=https://aws-external-anthropic.us-east-2.api.aws
export ANTHROPIC_WORKSPACE_ID=wrkspc_your_workspace_id
```

Notes:

- `LLM_PROVIDER=anthropic` is the only supported value;
  legacy values `openrouter` and `ollama` are coerced to `anthropic` with a warning.
- The per-agent provider variables
  (`ANNOTATION_PROVIDER`, `EVALUATION_PROVIDER`, `VISION_PROVIDER`, `LLM_PROVIDER_PREFERENCE`)
  were removed; models are selected by bare model identifier only.
- Set `ANNOTATION_MODEL=claude-sonnet-5` for the highest annotation quality.
- `HEDIT_PROMPT_CACHE_TTL` sets the prompt-cache lifetime, `5m` (default) or `1h`.
  Keep `5m` for a server; see [prompt-caching.md](../prompt-caching.md) for when `1h` is cheaper.

## API Usage

Once configured, the API works as follows:

```bash
curl -X POST http://localhost:38427/annotate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A red circle appears on the left side of the screen",
    "schema_version": "8.3.0",
    "max_validation_attempts": 3
  }'
```

## Bring Your Own Key (BYOK)

Users can supply their own first-party Anthropic API key (`sk-ant-...`)
via the `X-Anthropic-Key` request header.
BYOK requests go to `api.anthropic.com`, not the AWS workspace endpoint.
Model, evaluation-model, vision-model, and temperature overrides travel as
`X-Anthropic-Model`, `X-Anthropic-Eval-Model`, `X-Anthropic-Vision-Model`, and
`X-Anthropic-Temperature`.
The legacy `X-OpenRouter-*` spellings of those names remain accepted indefinitely,
so existing clients keep working.

## Troubleshooting

**Error: "ANTHROPIC_API_KEY environment variable is required"**
- Set the variable and reload your shell: `source ~/.bashrc`

**Requests rejected by the endpoint**
- Verify `ANTHROPIC_WORKSPACE_ID` is set; the endpoint rejects requests without the workspace header
- Verify `ANTHROPIC_BASE_URL` is `https://aws-external-anthropic.us-east-2.api.aws`

**HTTP 400 on model selection**
- Only `claude-haiku-4-5` and `claude-sonnet-5` are offered; other model identifiers are rejected

## Cost and Cache Reporting

Every annotation reports its token use, cost, and prompt-cache savings:
in the CLI output, in the `usage` field of API responses, and server-wide via
`GET /metrics` (server API key required).
The annotation prefix is ~21.8k tokens, so a cache hit removes roughly 80% of a
request's cost. See [prompt-caching.md](../prompt-caching.md) for measured figures
and the break-even math.

## Support

- Claude API documentation: https://docs.anthropic.com/
- HEDit issues: https://github.com/Annotation-Garden/HEDit/issues
