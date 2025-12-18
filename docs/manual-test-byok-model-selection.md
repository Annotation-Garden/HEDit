# Manual Test Plan: BYOK Model Selection

This document describes how to manually test the BYOK model/provider/temperature selection feature after PR #54 is merged.

## Prerequisites

1. Have a valid OpenRouter API key (get one at https://openrouter.ai)
2. Have the HEDit CLI installed: `pip install hedit` or `pip install -e .`
3. Know the API endpoint (e.g., `https://api.annotation.garden/hedit` or local `http://localhost:38427`)

## Test 1: Request Body Model Selection (API)

Test that model settings in the request body are used.

```bash
# Set your API key
export OPENROUTER_KEY="sk-or-v1-your-key-here"

# Test with custom model in request body
curl -X POST https://api.annotation.garden/hedit/annotate \
  -H "Content-Type: application/json" \
  -H "X-OpenRouter-Key: $OPENROUTER_KEY" \
  -d '{
    "description": "A red circle appears on the left side of the screen",
    "model": "openai/gpt-4o-mini",
    "temperature": 0.3
  }'
```

**Expected**: Should use `gpt-4o-mini` model (verify in OpenRouter dashboard usage logs).

## Test 2: Header-Based Model Selection (API)

Test that model settings in headers are used as fallback.

```bash
# Test with custom model in headers
curl -X POST https://api.annotation.garden/hedit/annotate \
  -H "Content-Type: application/json" \
  -H "X-OpenRouter-Key: $OPENROUTER_KEY" \
  -H "X-OpenRouter-Model: anthropic/claude-3-haiku-20240307" \
  -H "X-OpenRouter-Temperature: 0.1" \
  -d '{
    "description": "A blue square fades in at the center"
  }'
```

**Expected**: Should use `claude-3-haiku` model.

## Test 3: Request Body Overrides Headers

Test that request body has higher priority than headers.

```bash
curl -X POST https://api.annotation.garden/hedit/annotate \
  -H "Content-Type: application/json" \
  -H "X-OpenRouter-Key: $OPENROUTER_KEY" \
  -H "X-OpenRouter-Model: anthropic/claude-3-haiku-20240307" \
  -d '{
    "description": "A green triangle rotates",
    "model": "openai/gpt-4o-mini"
  }'
```

**Expected**: Should use `gpt-4o-mini` (body), NOT `claude-3-haiku` (header).

## Test 4: CLI Model Selection

Test the CLI with `--model` flag.

```bash
# Initialize with your key
hedit init --api-key $OPENROUTER_KEY

# Test with custom model
hedit annotate "A loud beep sound plays" --model openai/gpt-4o-mini --temperature 0.2
```

**Expected**: Should use specified model.

## Test 5: Image Annotation with Vision Model

Test image annotation with custom vision model.

```bash
# Create a test image or use any image file
curl -X POST https://api.annotation.garden/hedit/annotate-from-image \
  -H "Content-Type: application/json" \
  -H "X-OpenRouter-Key: $OPENROUTER_KEY" \
  -d "{
    \"image\": \"data:image/png;base64,$(base64 -i test_image.png)\",
    \"model\": \"openai/gpt-4o\",
    \"vision_model\": \"openai/gpt-4o\",
    \"temperature\": 0.3
  }"
```

**Expected**: Should use specified vision model for description.

## Test 6: Server Default Fallback

Test that without BYOK, server uses its defaults (this should already work).

```bash
# Using server API key (if you have one)
curl -X POST https://api.annotation.garden/hedit/annotate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-server-api-key" \
  -d '{
    "description": "A warning message appears"
  }'
```

**Expected**: Should use server's default model from environment variables.

## Test 7: Temperature Range Validation

Test that temperature validation works.

```bash
# Invalid temperature (should fail validation)
curl -X POST https://api.annotation.garden/hedit/annotate \
  -H "Content-Type: application/json" \
  -H "X-OpenRouter-Key: $OPENROUTER_KEY" \
  -d '{
    "description": "Test",
    "temperature": 1.5
  }'
```

**Expected**: Should return 422 validation error (temperature must be 0.0-1.0).

## Test 8: Provider Selection

Test provider preference (e.g., Cerebras for fast inference).

```bash
curl -X POST https://api.annotation.garden/hedit/annotate \
  -H "Content-Type: application/json" \
  -H "X-OpenRouter-Key: $OPENROUTER_KEY" \
  -d '{
    "description": "A participant presses a button",
    "model": "openai/gpt-oss-120b",
    "provider": "Cerebras"
  }'
```

**Expected**: Should route through Cerebras provider (faster inference).

## Verification

For all tests, verify:
1. The request succeeds (HTTP 200)
2. Valid HED annotation is returned
3. Check OpenRouter dashboard to confirm which model was used
4. Response time may vary by model/provider

## Notes

- The model parameter overrides ALL agents (annotation, evaluation, assessment)
- Per-agent model selection is not yet supported via API (future enhancement)
- Invalid model names will result in OpenRouter errors
