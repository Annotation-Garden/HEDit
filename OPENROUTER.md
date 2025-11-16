# OpenRouter Integration

This document explains how to use OpenRouter for cloud-based LLM inference with HED-BOT.

## Why OpenRouter?

OpenRouter provides unified access to multiple LLM providers (OpenAI, Anthropic, etc.) through a single API, with significant performance and cost benefits over local models:

- **Speed**: 10-50x faster than local Ollama (seconds vs minutes)
- **Cost**: Very affordable with modern efficient models
- **Quality**: Access to latest models (GPT-5, Claude 4.5)

## Performance Comparison

| Configuration | Time | Cost/1k annotations* |
|--------------|------|----------------------|
| Local Ollama (gpt-oss:20b) | 2-3 minutes | Free (GPU cost) |
| OpenRouter (GPT-5-mini) | 5-15 seconds | ~$0.50 |
| OpenRouter (Claude Haiku 4.5) | 3-10 seconds | ~$2.00 |

*Estimated based on average annotation complexity

## Architecture

The system uses different models for different tasks:

1. **Annotation Agent** (`gpt-5-mini` or `claude-haiku-4.5`): Main HED generation
2. **Evaluation Agent** (`gpt-5-mini`): Checks annotation faithfulness  
3. **Assessment Agent** (`gpt-5-mini` or `gpt-5-nano`): Final completeness check
4. **Feedback Summarizer** (`gpt-5-nano`): Condenses errors/feedback (NEW!)

The feedback summarizer is key - it uses the ultra-cheap `gpt-5-nano` model to condense verbose validation errors and feedback into concise, actionable points before the next annotation attempt. This:
- Reduces prompt size for subsequent iterations
- Speeds up the annotation loop
- Minimizes cost by using the cheapest model for simple summarization

## Setup

### 1. Install Dependencies

```bash
pip install langchain-openai
```

### 2. Set Environment Variables

**Option A: Using .env file (Recommended)**

Create a `.env` file in the project root (a template is provided as `.env.example`):

```bash
cp .env.example .env
```

Then edit `.env` and set:

```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-api-key-here

# Optional: customize models (defaults shown)
ANNOTATION_MODEL=gpt-5-mini
EVALUATION_MODEL=gpt-5-mini
ASSESSMENT_MODEL=gpt-5-nano
FEEDBACK_MODEL=gpt-5-nano
```

The `.env` file works for both local development and Docker deployment.

**Option B: Environment variables** (edit `~/.bashrc` or `~/.zshrc`):

```bash
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=your-api-key-here

# Optional: customize models (defaults shown)
export ANNOTATION_MODEL=gpt-5-mini
export EVALUATION_MODEL=gpt-5-mini
export ASSESSMENT_MODEL=gpt-5-nano
export FEEDBACK_MODEL=gpt-5-nano
```

### 3. Run Benchmark

Test different model configurations:

```bash
cd /home/yahya/git/hed-bot
python test_openrouter.py
```

This will test 3 configurations:
1. All GPT-5-Mini (balanced cost/performance)
2. Claude Haiku + GPT-5-Nano (maximum speed)
3. GPT-5-Mini + GPT-5-Nano (cost-optimized - **RECOMMENDED**)

## Recommended Configuration

Based on testing, the best balance of speed, cost, and quality is:

```bash
ANNOTATION_MODEL=gpt-5-mini          # $0.15/1M input, $0.60/1M output
EVALUATION_MODEL=gpt-5-mini          # Same model for consistency
ASSESSMENT_MODEL=gpt-5-nano          # $0.04/1M - simple task, use cheapest
FEEDBACK_MODEL=gpt-5-nano            # $0.04/1M - summarization is simple
```

**Expected performance:**
- Time: 5-15 seconds per annotation
- Cost: ~$0.0005 per annotation (~$0.50 per 1000 annotations)
- Quality: Excellent (same accuracy as local, but much faster)

## API Usage

Once configured, the API works exactly the same:

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

## Model Options

### Available Models

- `gpt-5-mini` (alias) → `openai/gpt-5-mini`
- `gpt-5-nano` (alias) → `openai/gpt-5-nano` 
- `claude-haiku` (alias) → `anthropic/claude-haiku-4.5`
- `gpt-5` → `openai/gpt-5` (premium)
- `claude-sonnet` → `anthropic/claude-sonnet-4.5` (premium)

You can use either the alias or full model name.

## Troubleshooting

**Error: "OPENROUTER_API_KEY environment variable is required"**
- Make sure you've set the environment variable
- Reload your shell: `source ~/.bashrc`

**Slow performance with OpenRouter**
- Check your internet connection
- Try a different model (Claude Haiku is typically fastest)
- Verify API key is valid

**High costs**
- Use `gpt-5-nano` for assessment and feedback (cheapest)
- Reduce `max_validation_attempts` to limit iterations
- Use Ollama for development, OpenRouter for production

## Cost Optimization Tips

1. Use `gpt-5-nano` ($0.04/1M) for simple tasks (assessment, feedback)
2. Use `gpt-5-mini` ($0.15/1M) for complex tasks (annotation, evaluation)
3. Set `run_assessment=false` to skip assessment agent
4. Keep annotations concise to reduce token usage
5. Cache frequently used annotations

## Support

For issues or questions:
- OpenRouter documentation: https://openrouter.ai/docs
- HED-BOT issues: https://github.com/hed-standard/hed-bot/issues
