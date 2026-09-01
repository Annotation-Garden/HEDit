# Contributing to HEDit

Thank you for your interest in contributing to HEDit! This guide covers our development workflow, issue management, and contribution guidelines.

## Issue and PR Labeling System

We use a structured labeling taxonomy to organize issues and pull requests. This helps with prioritization, filtering, and understanding the scope of work.

### Label Categories

#### Priority Labels

| Label | Description | Use When |
|-------|-------------|----------|
| `priority: critical` | Blocks release or causes data loss | Production bugs, security issues, data corruption |
| `priority: high` | Important for upcoming release | Key features for next version, significant bugs |
| `priority: medium` | Normal feature/fix | Standard features and improvements |
| `priority: low` | Nice to have | Future enhancements, minor improvements |

#### Type Labels

| Label | Description | Use When |
|-------|-------------|----------|
| `type: bug` | Something isn't working correctly | Unexpected behavior, errors, regressions |
| `type: feature` | New feature or enhancement | New functionality, improvements to existing features |
| `type: documentation` | Documentation improvements | README updates, guides, API docs |
| `type: performance` | Performance improvements | Speed optimizations, resource usage |
| `type: refactor` | Code refactoring without feature changes | Code cleanup, architectural improvements |

#### Component Labels

| Label | Description | Scope |
|-------|-------------|-------|
| `component: agents` | Related to LangGraph agents | Agent logic, prompts, workflows |
| `component: api` | Related to FastAPI backend | Endpoints, middleware, backend logic |
| `component: frontend` | Related to web frontend | UI, user experience, frontend code |
| `component: validation` | Related to HED validation | Validator integration, error handling |
| `component: cli` | Related to command-line interface | CLI commands, terminal UX |
| `component: ci-cd` | Related to CI/CD pipeline | GitHub Actions, testing, deployment automation |
| `component: deployment` | Related to deployment and infrastructure | Docker, cloud services, infrastructure |

#### Status Labels

| Label | Description | Use When |
|-------|-------------|----------|
| `status: in-progress` | Currently being worked on | Active development |
| `status: blocked` | Blocked by another issue or dependency | Waiting on external factors |
| `status: needs-discussion` | Needs discussion or design decisions | Requires team input |
| `status: needs-testing` | Needs testing or validation | Ready for QA |

#### Special Labels

| Label | Description |
|-------|-------------|
| `breaking change` | Contains breaking changes |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention is needed |
| `dependencies` | Updates a dependency |

### Labeling Guidelines

1. **Every issue should have at minimum:**
   - One `type:` label
   - One `priority:` label
   - One or more `component:` labels

2. **Multiple components are allowed** - An issue can span multiple components (e.g., `component: agents` + `component: validation` for a validation agent issue).

3. **Status labels are optional** - Only add when relevant to communicate blockers or progress.

4. **Update labels as work progresses** - Change priority if scope changes, add `status:` labels as needed.

### Examples

**Bug in validation logic:**
```
priority: high, type: bug, component: validation, component: api
```

**New agent feature for future version:**
```
priority: low, type: feature, component: agents
```

**CI/CD documentation update:**
```
priority: low, type: documentation, component: ci-cd
```

## Development Workflow

### Branch Naming

Use descriptive branch names:
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring

### Commit Messages

Write clear, concise commit messages:
- Use imperative mood ("Add feature" not "Added feature")
- Keep first line under 72 characters
- Reference issues when relevant (`Fixes #123`)

### Pull Requests

1. Reference related issues in the PR description
2. Include a summary of changes
3. Add a test plan for reviewers
4. Request review from maintainers

## Code Style

- **Python**: Follow PEP 8, enforced by Ruff pre-commit hooks
- **JavaScript**: Standard ES6+ conventions
- **Testing**: Use pytest with coverage, prefer integration tests over mocks

## Getting Help

- Check existing issues before creating new ones
- Use GitHub Discussions for questions
- Tag issues appropriately for visibility
