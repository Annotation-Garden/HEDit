# GitHub Security Scanning Setup

This repository includes comprehensive security scanning workflows. Some features require GitHub Advanced Security (GHAS).

## Current Status

### ✅ Always Available (Free Tier)
These security scans work on all repositories:

- **Dependabot Alerts** - Notifies of vulnerable dependencies
- **Python Security (Safety)** - Checks known vulnerability databases
- **Python Security (Bandit)** - Static security analysis
- **Docker Security (Trivy)** - Scans Docker images for vulnerabilities
- **Secret Scanning** (TruffleHog) - Detects leaked secrets

### ⚠️ Requires GitHub Advanced Security
These features need GHAS or public repository:

- **CodeQL Analysis** - Advanced code security scanning
- **Dependency Review** - Automated dependency vulnerability review
- **Security Tab Integration** - SARIF upload for Trivy/CodeQL results

## How to Enable Full Security Features

### Option 1: Make Repository Public (Recommended)

**Free** and enables all security features automatically:

1. Go to **Settings → General → Danger Zone**
2. Click **Change visibility → Make public**
3. All security features activate automatically
4. CodeQL and dependency scanning work immediately

**Benefits:**
- ✅ All security features FREE
- ✅ No configuration needed
- ✅ Community can contribute
- ✅ Better for open source project

### Option 2: Enable GitHub Advanced Security (Private Repos)

**Paid feature** for private repositories:

1. Repository must be part of GitHub Enterprise Cloud or GitHub Enterprise Server
2. Go to **Settings → Code security and analysis**
3. Enable **GitHub Advanced Security**
4. Enable **CodeQL analysis**
5. Enable **Dependency review**

**Cost:**
- Requires GitHub Enterprise subscription
- Billed per active committer
- See: https://docs.github.com/en/billing/managing-billing-for-github-advanced-security

### Option 3: Keep as Private (Current)

The workflows are configured to:
- ✅ Run all free security scans
- ✅ Upload reports as artifacts (downloadable)
- ⚠️ Skip GHAS features gracefully (no errors)
- ⚠️ No Security tab integration

**Security reports available in:**
- Actions → Security Scan → Artifacts
  - `bandit-security-report.json`
  - `trivy-security-report.sarif`

## Enabling Specific Features

### Enable Dependabot

1. Go to **Settings → Code security and analysis**
2. Enable **Dependency graph**
3. Enable **Dependabot alerts**
4. Enable **Dependabot security updates**

This is **FREE** for all repositories!

### Enable Secret Scanning (Push Protection)

**Free for public repos**, requires GHAS for private:

1. Go to **Settings → Code security and analysis**
2. Enable **Secret scanning**
3. Enable **Push protection** (prevents committing secrets)

### Enable CodeQL

**Automatic on first workflow run** (public repos only):

1. Merge the PR with CodeQL workflow
2. First run enables it automatically
3. Results appear in **Security → Code scanning**

For private repos: Requires GHAS

## Workflow Behavior

### CodeQL Workflow
```yaml
# Runs on: Public repos or repos with GHAS
if: github.event.repository.visibility == 'public'
```

**Private repo without GHAS**: Workflow is skipped (no error)
**Public repo**: Runs automatically
**Private repo with GHAS**: Runs automatically

### Security Scan Workflow
```yaml
# Trivy SARIF upload
if: github.event.repository.visibility == 'public'
```

**Always runs**: Trivy scan, Bandit, Safety, TruffleHog
**Conditional**: SARIF upload to Security tab (public only)
**Alternative**: Results available as downloadable artifacts

## Viewing Security Results

### With GHAS Enabled
- **Security tab** → Code scanning alerts
- **Security tab** → Dependabot alerts
- **Security tab** → Secret scanning alerts

### Without GHAS (Current)
- **Actions** → Security Scan → Artifacts
  - Download `bandit-security-report.json`
  - Download `trivy-security-report.sarif`
- **Pull Requests** → Checks → Security Scan logs

## Recommendations

### For Open Source Projects
**Make repository public** - All features FREE and automatic

### For Private Projects
**Current setup is fine** - Core security scanning still works:
- ✅ Dependabot alerts enabled
- ✅ Python security scans (Bandit, Safety)
- ✅ Docker security scans (Trivy)
- ✅ Secret detection (TruffleHog)
- ✅ Reports in artifacts

**Optional:** Upgrade to GHAS if budget allows

## Current Workflow Status

| Workflow | Status | Notes |
|----------|--------|-------|
| **Dependabot** | ✅ Active | Creates PRs for vulnerabilities |
| **Security Scan** | ✅ Active | All scans run, artifacts available |
| **CodeQL** | ⏸️ Skipped | Waiting for public or GHAS |
| **Docker Build** | ✅ Active | Builds and tests images |

## Troubleshooting

### Error: "Code scanning is not enabled"
**Cause**: CodeQL trying to upload results without GHAS
**Fix**: Workflows now skip gracefully (no error)
**Alternative**: Make repo public or enable GHAS

### Error: "Resource not accessible by integration"
**Cause**: Trying to write to Security tab without permissions
**Fix**: Workflows now check repo visibility first
**Workaround**: Download artifacts instead

### Want Security Tab Results?
**Option 1**: Make repository public (free)
**Option 2**: Enable GHAS (paid)
**Option 3**: Download artifacts from workflow runs

## Summary

**Current Setup:**
- ✅ Core security scanning works
- ✅ No workflow failures
- ✅ Reports available as artifacts
- ⚠️ No Security tab integration (needs GHAS or public)

**To Enable Full Features:**
- Make repository public (recommended, free)
- Or enable GitHub Advanced Security (paid)

**Everything is configured and ready** - just enable GHAS or make public when ready!

---

**Questions?** See:
- [GitHub Advanced Security docs](https://docs.github.com/en/get-started/learning-about-github/about-github-advanced-security)
- [Code scanning docs](https://docs.github.com/en/code-security/code-scanning)
- Repository SECURITY.md for vulnerability reporting
