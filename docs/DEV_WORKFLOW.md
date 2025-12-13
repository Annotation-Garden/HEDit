# Development Workflow

This document describes the development and testing workflow for hed-bot.

## Quick Reference

| Level | Use Case | Command |
|-------|----------|---------|
| Unit Tests | Test specific modules | `python scripts/test_error_remediation.py` |
| Local API | Full backend locally | `./scripts/dev.sh` |
| API Tests | Test running server | `python scripts/test_api.py` |
| Dev Channel | Staging environment | Push to `develop` branch |

## Level 1: Unit Testing (Fastest)

Test specific modules without running the full backend:

```bash
# Test error remediation
python scripts/test_error_remediation.py

# Run pytest (when environment is set up)
pytest tests/ -v
```

## Level 2: Local Backend

Run the full backend locally for integration testing:

```bash
# Start local dev server
./scripts/dev.sh

# With no authentication (easier testing)
./scripts/dev.sh --no-auth

# In another terminal, test the API
python scripts/test_api.py
```

### Prerequisites

1. **Environment file**: Copy `.env.example` to `.env` and configure:
   ```bash
   LLM_PROVIDER=openrouter
   OPENROUTER_API_KEY=your-key
   ```

2. **HED repositories**: Clone these locally:
   ```bash
   # Schema files
   git clone https://github.com/hed-standard/hed-schemas ~/Documents/git/HED/hed-schemas

   # JavaScript validator
   git clone https://github.com/hed-standard/hed-javascript ~/Documents/git/HED/hed-javascript
   cd ~/Documents/git/HED/hed-javascript && npm install && npm run build
   ```

## Level 3: Dev Channel (Staging)

For testing changes before production, use the develop branch:

### Workflow

```
feature/xxx → develop → main
     ↓           ↓        ↓
  (local)     (dev)   (prod)
```

### How it works

1. **Create feature branch** from main
2. **Test locally** using Level 1 & 2
3. **Push to develop** for staging deployment
4. **Merge to main** for production

### Dev environment

- Docker image: `ghcr.io/neuromechanist/hed-bot:develop`
- API endpoint: `hed-bot-dev-api.workers.dev` (when configured)
- Frontend: Cloudflare Pages preview deployments

## Testing the Error Remediation Feature

The error remediation feature adds actionable guidance to validation errors.

### Test Cases

1. **TAG_EXTENDED warning** (extension from schema):
   ```
   Description: "A red house appears on screen"
   Expected: Warning about Building/House vs Item/House
   ```

2. **TAG_EXTENSION_INVALID** (extending existing tag):
   ```
   Force annotation with: Property/Red
   Expected: Error with guidance to use "Red" directly
   ```

3. **DEFINITION_INVALID** (malformed definition):
   ```
   HED string: "Definition/MyDef"
   Expected: Error with correct pattern
   ```

### Verification

```bash
# Quick module test
python scripts/test_error_remediation.py

# Full API test (with running server)
python scripts/test_api.py --description "A red house appears on screen"
```

Look for `REMEDIATION` in the output to confirm the feature is working.

## Troubleshooting

### Server won't start

1. Check `.env` file exists with valid API keys
2. Verify HED schema/validator paths exist
3. Check port 38427 is not in use

### No remediation in output

1. Annotation may be fully valid (no errors to remediate)
2. Try a description that triggers extensions: "A house appears"
3. Check validation_errors and validation_warnings in response

### LLM errors

1. Verify OPENROUTER_API_KEY is set
2. Check API key has credits
3. Try with `--no-auth` to rule out auth issues
