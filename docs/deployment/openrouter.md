# OpenRouter Integration

This document explains how to use OpenRouter for cloud-based LLM inference with HED-BOT.

## Why OpenRouter?

OpenRouter provides unified access to multiple LLM providers through a single API, with significant performance benefits:

- **Speed**: Ultra-fast inference via Cerebras provider
- **Quality**: Access to capable open-source models
- **Flexibility**: Easy model switching without code changes

## Architecture

The system uses different models for different tasks:

1. **Annotation Agent** (`openai/gpt-oss-120b` via Cerebras): Main HED generation
2. **Evaluation Agent** (`qwen/qwen3-235b-a22b-2507`): Checks annotation faithfulness
3. **Assessment Agent** (`openai/gpt-oss-120b` via Cerebras): Final completeness check
4. **Feedback Summarizer** (`openai/gpt-oss-120b` via Cerebras): Condenses errors/feedback

The `gpt-oss-120b` model via Cerebras provides extremely fast inference, making the annotation workflow responsive.

## Setup

### 1. Install Dependencies

```bash
pip install langchain-openai
```

### 2. Set Environment Variables

**Option A: Using .env file (Recommended)**

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-api-key-here
LLM_PROVIDER_PREFERENCE=Cerebras

# Model configuration (defaults shown)
ANNOTATION_MODEL=openai/gpt-oss-120b
EVALUATION_MODEL=qwen/qwen3-235b-a22b-2507
ASSESSMENT_MODEL=openai/gpt-oss-120b
FEEDBACK_MODEL=openai/gpt-oss-120b
```

**Option B: Environment variables** (edit `~/.bashrc` or `~/.zshrc`):

```bash
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=your-api-key-here
export LLM_PROVIDER_PREFERENCE=Cerebras

export ANNOTATION_MODEL=openai/gpt-oss-120b
export EVALUATION_MODEL=qwen/qwen3-235b-a22b-2507
export ASSESSMENT_MODEL=openai/gpt-oss-120b
export FEEDBACK_MODEL=openai/gpt-oss-120b
```

## API Usage

Once configured, the API works as follows:

```bash
curl -X POST http://localhost:38427/annotate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A red circle appears on the left side of the screen",
    "schema_version": "8.3.0",
    "max_validation_attempts": 3,
    "run_assessment": false
  }'
```

The system automatically uses OpenRouter when `LLM_PROVIDER=openrouter`.

## Switching Back to Ollama

To use local Ollama again:

```bash
export LLM_PROVIDER=ollama
# OR remove the environment variable entirely (defaults to Ollama)
```

## Model Configuration

### Current Models

| Agent | Model | Provider |
|-------|-------|----------|
| Annotation | `openai/gpt-oss-120b` | Cerebras |
| Evaluation | `qwen/qwen3-235b-a22b-2507` | Cerebras |
| Assessment | `openai/gpt-oss-120b` | Cerebras |
| Feedback | `openai/gpt-oss-120b` | Cerebras |

### Provider Preference

The `LLM_PROVIDER_PREFERENCE=Cerebras` setting routes all model requests through Cerebras for ultra-fast inference.

## Troubleshooting

**Error: "OPENROUTER_API_KEY environment variable is required"**
- Make sure you've set the environment variable
- Reload your shell: `source ~/.bashrc`

**Error: "No cookie auth credentials found"**
- Check that your API key is valid
- Verify the model is available on OpenRouter
- Ensure the provider supports the requested model

**Slow performance**
- Check your internet connection
- Verify `LLM_PROVIDER_PREFERENCE=Cerebras` is set for fast inference

## Support

For issues or questions:
- OpenRouter documentation: https://openrouter.ai/docs
- HED-BOT issues: https://github.com/hed-standard/hed-bot/issues
