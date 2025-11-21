# Version Management

HED-BOT follows [Semantic Versioning 2.0.0](https://semver.org/).

## Current Version

**0.3.0-alpha**

## Version Format

```
MAJOR.MINOR.PATCH[-PRERELEASE]
```

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)
- **PRERELEASE**: Optional pre-release label
  - `alpha`: Early testing, may have significant bugs
  - `beta`: Feature-complete, testing and refinement
  - `rc`: Release candidate, final testing before stable
  - Omitted for stable releases (e.g., `1.0.0`)

## Bumping Version

Use the version bump script to increment the version:

```bash
# Patch version bump (0.3.0-alpha -> 0.3.1-alpha)
python scripts/bump_version.py patch

# Minor version bump (0.3.0-alpha -> 0.4.0-alpha)
python scripts/bump_version.py minor

# Major version bump (0.3.0-alpha -> 1.0.0-alpha)
python scripts/bump_version.py major

# Change pre-release label
python scripts/bump_version.py patch --prerelease beta   # -> 0.3.1-beta
python scripts/bump_version.py minor --prerelease stable # -> 0.4.0 (no label)

# Show current version
python scripts/bump_version.py --current

# Skip git operations (for testing)
python scripts/bump_version.py patch --no-git
```

## Workflow

### 1. Development in Feature Branch

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "Add new feature"
```

### 2. Bump Version After PR Merge

After your PR is merged to `main`:

```bash
# Switch to main and pull latest
git checkout main
git pull origin main

# Bump version (choose appropriate bump type)
python scripts/bump_version.py patch  # or minor/major

# Push commit and tag
git push origin main
git push origin v0.3.1-alpha  # replace with actual version
```

### 3. Automatic GitHub Release

When you push a version tag (e.g., `v0.3.1-alpha`), GitHub Actions automatically:

1. Creates a GitHub Release
2. Generates changelog from commits
3. Marks pre-release appropriately (alpha, beta, rc)
4. Adds installation instructions

## Version Display

The version is displayed in:

1. **Frontend Footer**: Visible at the bottom of the web interface
2. **API Endpoints**:
   - `GET /version` - Returns version JSON
   - `GET /health` - Includes version in health check
   - `GET /` - Root endpoint shows version
3. **Feedback Reports**: Version included in feedback emails

## Version History

| Version | Release Date | Description |
|---------|-------------|-------------|
| 0.3.0-alpha | 2025-01-XX | Add semantic versioning system |
| 0.2.0-alpha | 2025-01-XX | Badge tooltips and validation fixes |
| 0.1.0 | 2025-01-XX | Initial release |

## Pre-release Guidelines

- **Alpha**: Active development, expect changes
  - Use for early feature development
  - May have incomplete features
  - Breaking changes allowed

- **Beta**: Feature freeze, bug fixing
  - Use when features are complete
  - Focus on testing and stability
  - Avoid breaking changes unless critical

- **RC (Release Candidate)**: Final testing
  - Use before stable release
  - No new features
  - Only critical bug fixes

- **Stable** (no label): Production ready
  - Thoroughly tested
  - Documentation complete
  - Breaking changes require major version bump

## Notes

- Always bump version **after** PR merge, not before
- Keep version bumps in separate commits
- Don't manually edit `src/version.py` - use the bump script
- Tag format must be `vX.Y.Z` or `vX.Y.Z-prerelease`
- Pre-release versions are marked accordingly on GitHub Releases
