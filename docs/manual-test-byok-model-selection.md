# Manual Test Plan: BYOK Model Selection

This document describes how to manually test the bring your own key (BYOK) model/temperature selection feature.

## Prerequisites

1. Have a valid first-party Anthropic API key (`sk-ant-...`, get one at https://console.anthropic.com).
   BYOK requests go to `api.anthropic.com`, not the server's Claude Platform on AWS workspace.
2. Have the HEDit CLI installed: `pip install hedit` or `pip install -e .`
3. Know the API endpoint (e.g., `https://api.annotation.garden/hedit` or local `http://localhost:38427`)

Only two models are offered: `claude-haiku-4-5` (default) and `claude-sonnet-5` (highest quality).
Legacy OpenRouter-style identifiers such as `anthropic/claude-haiku-4.5` are accepted as aliases.

## Test 1: Request Body Model Selection (API)

Test that model settings in the request body are used.

```bash
# Set your API key
export ANTHROPIC_KEY="sk-ant-your-key-here"

# Test with custom model in request body
curl -X POST https://api.annotation.garden/hedit/annotate \
  -H "Content-Type: application/json" \
  -H "X-Anthropic-Key: $ANTHROPIC_KEY" \
  -d '{
    "description": "A red circle appears on the left side of the screen",
    "model": "claude-sonnet-5",
    "temperature": 0.3
  }'
```

**Expected**: Should use `claude-sonnet-5` (verify in the Anthropic Console usage logs).

## Test 2: Header-Based Model Selection (API)

Test that model settings in headers are used as fallback.
The override headers keep their legacy names.

```bash
# Test with custom model in headers
curl -X POST https://api.annotation.garden/hedit/annotate \
  -H "Content-Type: application/json" \
  -H "X-Anthropic-Key: $ANTHROPIC_KEY" \
  -H "X-OpenRouter-Model: claude-sonnet-5" \
  -H "X-OpenRouter-Temperature: 0.1" \
  -d '{
    "description": "A blue square fades in at the center"
  }'
```

**Expected**: Should use `claude-sonnet-5`.

## Test 3: Request Body Overrides Headers

Test that request body has higher priority than headers.

```bash
curl -X POST https://api.annotation.garden/hedit/annotate \
  -H "Content-Type: application/json" \
  -H "X-Anthropic-Key: $ANTHROPIC_KEY" \
  -H "X-OpenRouter-Model: claude-sonnet-5" \
  -d '{
    "description": "A green triangle rotates",
    "model": "claude-haiku-4-5"
  }'
```

**Expected**: Should use `claude-haiku-4-5` (body), NOT `claude-sonnet-5` (header).

## Test 4: CLI Model Selection

Test the CLI with `--model` flag.

```bash
# Store your BYOK key (optional; HEDIT_ANTHROPIC_API_KEY env var also works)
hedit init --api-key $ANTHROPIC_KEY

# Test with custom model
hedit annotate "A loud beep sound plays" --model claude-sonnet-5 --temperature 0.2
```

**Expected**: Should use specified model.

## Test 5: Image Annotation with Vision Model

Test image annotation with a custom vision model.

```bash
# Create a test image or use any image file
curl -X POST https://api.annotation.garden/hedit/annotate-from-image \
  -H "Content-Type: application/json" \
  -H "X-Anthropic-Key: $ANTHROPIC_KEY" \
  -d "{
    \"image\": \"data:image/png;base64,$(base64 -i test_image.png)\",
    \"model\": \"claude-sonnet-5\",
    \"vision_model\": \"claude-sonnet-5\",
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

**Expected**: Should use server's default model (`claude-haiku-4-5`) from environment variables,
billed through the server's Claude Platform on AWS workspace.

## Test 7: Temperature Range Validation

Test that temperature validation works.

```bash
# Invalid temperature (should fail validation)
curl -X POST https://api.annotation.garden/hedit/annotate \
  -H "Content-Type: application/json" \
  -H "X-Anthropic-Key: $ANTHROPIC_KEY" \
  -d '{
    "description": "Test",
    "temperature": 1.5
  }'
```

**Expected**: Should return 422 validation error (temperature must be 0.0-1.0).

## Test 8: Non-Claude Model Rejection

Test that unsupported model identifiers are rejected.

```bash
curl -X POST https://api.annotation.garden/hedit/annotate \
  -H "Content-Type: application/json" \
  -H "X-Anthropic-Key: $ANTHROPIC_KEY" \
  -d '{
    "description": "A participant presses a button",
    "model": "qwen/qwen3-235b-a22b-2507"
  }'
```

**Expected**: Should return HTTP 400
(`qwen`, `mistral`, and `gpt-oss` identifiers are rejected;
only `claude-haiku-4-5` and `claude-sonnet-5` are offered).

## Test 9: Legacy BYOK Header Transport

Test that the legacy `X-OpenRouter-Key` header is still accepted as transport
when it carries an `sk-ant` key.

```bash
curl -X POST https://api.annotation.garden/hedit/annotate \
  -H "Content-Type: application/json" \
  -H "X-OpenRouter-Key: $ANTHROPIC_KEY" \
  -d '{
    "description": "A tone plays through the left speaker"
  }'
```

**Expected**: Should succeed exactly like `X-Anthropic-Key`.
Legacy provider headers
(`X-OpenRouter-Provider`, `X-OpenRouter-Eval-Provider`, `X-OpenRouter-Vision-Provider`)
are accepted but ignored.

## Verification

For all tests, verify:
1. The request succeeds (HTTP 200), except the rejection tests (7 and 8)
2. Valid HED annotation is returned
3. Check the Anthropic Console dashboard to confirm which model was used
4. Response time may vary by model

## Notes

- The model parameter overrides ALL agents (annotation, evaluation, assessment);
  the evaluation model can be set separately via the `X-OpenRouter-Eval-Model` header
- BYOK requests are billed to your own Anthropic account, not the server's AWS workspace
- Model identifiers other than the two offered Claude models result in HTTP 400
