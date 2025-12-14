# Security Policy

## Supported Versions

We release security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

## Reporting a Vulnerability

We take the security of HED-BOT seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Reporting Process

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories** (Preferred)
   - Go to the repository's [Security tab](https://github.com/neuromechanist/hed-bot/security/advisories)
   - Click "Report a vulnerability"
   - Fill out the form with details

2. **Email** (Alternative)
   - Send an email to the repository maintainers
   - Include "SECURITY" in the subject line
   - Provide detailed information about the vulnerability

### What to Include

Please include the following information in your report:

- Type of vulnerability (e.g., XSS, SQL injection, authentication bypass)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours of report submission
- **Vulnerability Confirmation**: Within 7 days of report
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next scheduled release

### Disclosure Policy

- Security issues will be publicly disclosed only after a fix has been released
- We will credit reporters who responsibly disclose vulnerabilities (unless they prefer to remain anonymous)
- We follow coordinated vulnerability disclosure principles

## Security Features

HED-BOT implements the following security measures:

### Authentication & Authorization
- API key authentication for all protected endpoints
- Multiple API key support for key rotation
- Optional authentication bypass for development

### Audit Logging
- Complete request/response audit trail
- IP address logging
- API key usage tracking
- 90-day log retention

### CORS Protection
- Strict origin validation
- Whitelist-based origin control
- Support for development environments

### Security Headers
- `Strict-Transport-Security` (HSTS)
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`

### Input Validation
- Pydantic model validation for all inputs
- Schema-based validation
- Request size limits

### Rate Limiting
- Per-IP rate limiting (configurable)
- Burst protection
- 429 status code for rate limit exceeded

### Secrets Management
- Environment variable-based configuration
- No secrets in Git repository
- .env files in .gitignore
- Secrets scanning enabled

### Dependency Security
- Dependabot automated updates
- CodeQL security scanning
- Regular dependency audits

## Security Best Practices for Deployers

When deploying HED-BOT:

1. **Always use HTTPS** in production
2. **Enable API key authentication** (`REQUIRE_API_AUTH=true`)
3. **Enable audit logging** (`ENABLE_AUDIT_LOG=true`)
4. **Restrict CORS origins** to only your frontend domain
5. **Set file permissions** on .env files (`chmod 600 .env`)
6. **Rotate API keys** quarterly or after any suspected compromise
7. **Monitor audit logs** for suspicious activity
8. **Keep dependencies updated** (Dependabot will help)
9. **Use rate limiting** to prevent abuse
10. **Review security headers** in Nginx/reverse proxy

See [`deploy/SECURITY.md`](deploy/SECURITY.md) for complete security documentation.

## Known Security Considerations

### LLM API Keys
- OpenRouter API keys are stored as environment variables
- Keys are passed to OpenRouter API via HTTPS
- Keys are not logged in audit logs
- Rotate keys if compromised

### Cloudflare Worker Proxy
- Worker authenticates to backend with API key
- API key stored as Cloudflare Worker secret
- Not visible in frontend JavaScript
- Recommended for production deployments

### Local GPU Deployment
- Ollama runs locally (no API key needed)
- No external API calls for LLM inference
- Complete privacy for offline operation
- Ensure GPU drivers are up to date

## Security Scanning

This repository uses:

- **Dependabot**: Automated dependency updates
- **CodeQL**: Static code analysis for security issues
- **Secret Scanning**: Detects accidentally committed secrets
- **Dependency Review**: Reviews dependency changes in PRs

## Compliance

HED-BOT is designed to comply with:

- **OWASP Top 10 (2021)**: See `deploy/SECURITY.md` for compliance mapping
- **Security Audit Requirements**: Complete audit logging and access controls
- **API Security Best Practices**: Authentication, rate limiting, CORS

## Security Contacts

For security-related questions or concerns:

- **Security Issues**: Use GitHub Security Advisories
- **General Questions**: Open a GitHub issue with `security` label
- **Private Concerns**: Contact repository maintainers directly

---

**Last Updated**: December 2, 2025
