# Development Workflow

This document describes the development and testing workflow for hed-bot.

## Branch Strategy

```
feature/* ──PR──> develop ──PR──> main
    │                │              │
 (local)         (staging)       (prod)
                    │              │
              :develop tag    :latest tag
                    │              │
              dev tunnel      prod tunnel
```

- **feature/\***: Development branches for new features/fixes
- **develop**: Staging environment for integration testing
- **main**: Production environment

## Quick Reference

| Level | Use Case | Command |
|-------|----------|---------|
| Unit Tests | Test specific modules | `python scripts/test_error_remediation.py` |
| Local API | Full backend locally | `./scripts/dev.sh` |
| API Tests | Test running server | `python scripts/test_api.py` |
| Staging | Integration testing | PR to `develop` branch |
| Production | Release | PR from `develop` to `main` |

## Level 1: Unit Testing (Fastest)

Test specific modules without running the full backend:

```bash
# Test error remediation
python scripts/test_error_remediation.py

# Run pytest (when environment is set up)
pytest tests/ -v
```

## Level 2: Local Backend

Run the full backend locally for integration testing.

### Prerequisites

1. **Environment file**: Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your OPENROUTER_API_KEY
   ```

2. **HED repositories**: Clone these locally:
   ```bash
   # Schema files
   git clone https://github.com/hed-standard/hed-schemas ~/Documents/git/HED/hed-schemas

   # JavaScript validator
   git clone https://github.com/hed-standard/hed-javascript ~/Documents/git/HED/hed-javascript
   cd ~/Documents/git/HED/hed-javascript && npm install && npm run build
   ```

### Running Local Server

```bash
# Start local dev server
./scripts/dev.sh

# With no authentication (easier testing)
./scripts/dev.sh --no-auth
```

### Testing the API

```bash
# In another terminal
python scripts/test_api.py

# With custom description
python scripts/test_api.py --description "A red house appears on screen"

# With API key (if auth enabled)
python scripts/test_api.py --api-key your-key
```

## Level 3: Staging (develop branch)

After local testing passes, create a PR to the `develop` branch.

### Workflow

1. **Create feature branch** from main:
   ```bash
   git checkout main
   git pull
   git checkout -b feature/my-feature
   ```

2. **Develop and test locally** (Level 1 & 2)

3. **Push and create PR to develop**:
   ```bash
   git push -u origin feature/my-feature
   gh pr create --base develop
   ```

4. **Staging deployment**:
   - Docker image built: `ghcr.io/neuromechanist/hed-bot:develop`
   - Dev backend pulls and runs the image
   - Test via dev API endpoint

5. **After staging validation**, create PR from develop to main

### Staging Environment

| Component | URL/Location |
|-----------|--------------|
| Docker Image | `ghcr.io/neuromechanist/hed-bot:develop` |
| Dev API | `hed-bot-dev-api.workers.dev` (when configured) |
| Frontend Preview | Cloudflare Pages preview deployments |

## Level 4: Production (main branch)

After staging validation, merge develop to main.

```bash
gh pr create --base main --head develop --title "Release: feature description"
```

### Production Environment

| Component | URL/Location |
|-----------|--------------|
| Docker Image | `ghcr.io/neuromechanist/hed-bot:latest` |
| Prod API | `hed-bot-api.workers.dev` |
| Frontend | `hed-bot.pages.dev` |

## Testing the Error Remediation Feature

The error remediation feature adds actionable guidance to validation errors.

### Test Cases

1. **TAG_EXTENDED warning** (extension from schema):
   ```
   Description: "A red house appears on screen"
   Expected: Warning about Building/House vs Item/House with remediation
   ```

2. **TAG_EXTENSION_INVALID** (extending existing tag):
   ```
   Force annotation with: Property/Red
   Expected: Error with guidance to use "Red" directly
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

1. Check `.env` file exists with valid `OPENROUTER_API_KEY`
2. Verify HED schema/validator paths exist
3. Check port 38427 is not in use: `lsof -i :38427`

### No remediation in output

1. Annotation may be fully valid (no errors to remediate)
2. Try a description that triggers extensions: "A house appears"
3. Check `validation_errors` and `validation_warnings` in response

### LLM errors

1. Verify `OPENROUTER_API_KEY` is set in `.env`
2. Check API key has credits at openrouter.ai
3. Check network connectivity
