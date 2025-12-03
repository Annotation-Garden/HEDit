# HED-BOT Security Best Practices

This document outlines the security measures implemented in HED-BOT to ensure compliance with security audits and best practices.

## Table of Contents

- [Overview](#overview)
- [Authentication & Authorization](#authentication--authorization)
- [CORS & Origin Validation](#cors--origin-validation)
- [Audit Logging](#audit-logging)
- [Security Headers](#security-headers)
- [Rate Limiting](#rate-limiting)
- [HTTPS & Encryption](#https--encryption)
- [Environment Variables](#environment-variables)
- [Audit Compliance](#audit-compliance)

---

## Overview

HED-BOT implements defense-in-depth security with multiple layers:

1. **API Key Authentication** - Prevents unauthorized access
2. **CORS Validation** - Allows only approved origins
3. **Audit Logging** - Complete request/response trail
4. **Security Headers** - Protects against common attacks
5. **Rate Limiting** - Prevents abuse and DoS
6. **HTTPS Only** - Encrypted communication
7. **Input Validation** - Prevents injection attacks

---

## Authentication & Authorization

### API Key Authentication

**Implementation**: FastAPI dependency injection with custom middleware

**Location**: `src/api/security.py`

#### How It Works

1. Client includes API key in `X-API-Key` header
2. FastAPI validates key before processing request
3. Invalid/missing keys return `401 Unauthorized`
4. All authentication events are audit logged

#### Generating API Keys

```bash
# Generate a secure random API key
python scripts/generate_api_key.py

# Output: API Key: a1b2c3d4e5f6...  (64 characters)
```

#### Configuring API Keys

**Option 1: Environment Variable (Recommended)**

```bash
# .env file
API_KEYS=key1_64_chars,key2_64_chars,key3_64_chars
```

**Option 2: Individual Keys**

```bash
# .env file
API_KEY_1=first_key_64_chars
API_KEY_2=second_key_64_chars
API_KEY_3=third_key_64_chars
```

#### Disabling Authentication (Development Only)

```bash
# .env file
REQUIRE_API_AUTH=false  # NOT recommended for production
```

⚠️ **Warning**: Never disable authentication in production!

#### Using API Keys

**Frontend (JavaScript)**:
```javascript
fetch('https://hedtools.ucsd.edu/hed-bot-api/annotate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your_api_key_here',
  },
  body: JSON.stringify({ description: '...' }),
});
```

**cURL**:
```bash
curl -X POST https://hedtools.ucsd.edu/hed-bot-api/annotate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{"description": "person sees red circle"}'
```

**Python**:
```python
import requests

response = requests.post(
    'https://hedtools.ucsd.edu/hed-bot-api/annotate',
    headers={'X-API-Key': 'your_api_key_here'},
    json={'description': 'person sees red circle'}
)
```

### Protected vs Public Endpoints

**Protected (Require API Key)**:
- `POST /annotate` - Generate annotations
- `POST /annotate-from-image` - Image annotations
- `POST /annotate/stream` - Streaming annotations
- `POST /validate` - Validate HED strings

**Public (No API Key)**:
- `GET /health` - Health checks (for monitoring)
- `GET /version` - Version information
- `GET /` - API documentation

---

## CORS & Origin Validation

### Allowed Origins

**Production**: Only `https://hed-bot.pages.dev`

**Development**: `http://localhost:5173`, `http://localhost:3000`

### Configuration

```python
# src/api/main.py
allowed_origins = [
    "https://hed-bot.pages.dev",  # Production
    "http://localhost:5173",       # Dev
]
```

### Adding Extra Origins

```bash
# .env file
EXTRA_CORS_ORIGINS=https://staging.hed-bot.pages.dev,https://dev.hed-bot.pages.dev
```

### Two-Layer CORS Protection

1. **Nginx Layer**: Validates Origin header
2. **FastAPI Layer**: Enforces CORS policy

Even if Nginx is bypassed, FastAPI will reject invalid origins.

---

## Audit Logging

### What Gets Logged

**Every Request**:
- Timestamp (ISO 8601 format)
- Client IP address
- HTTP method and path
- API key hash (first 8 characters)
- User identifier (if available)

**Every Response**:
- HTTP status code
- Processing time (milliseconds)
- Response size (if applicable)

**Errors**:
- Error type and message
- Stack trace (in debug mode)
- Associated request details

### Log Format

```
2025-12-02T15:30:45.123Z - hed_bot.audit - INFO - [AUDIT] REQUEST - timestamp=2025-12-02T15:30:45.123Z, ip=1.2.3.4, method=POST, path=/annotate, api_key=a1b2c3d4..., user=anonymous
2025-12-02T15:30:47.456Z - hed_bot.audit - INFO - [AUDIT] RESPONSE - timestamp=2025-12-02T15:30:47.456Z, ip=1.2.3.4, method=POST, path=/annotate, status=200, duration_ms=2333.45
```

### Log Locations

**Audit Log**: `/var/log/hed-bot/audit.log`
**Application Log**: Docker container logs (via `docker logs`)
**Nginx Access Log**: `/var/log/nginx/access.log`
**Nginx Error Log**: `/var/log/nginx/error.log`

### Configuration

```bash
# .env file
ENABLE_AUDIT_LOG=true  # Enable/disable audit logging
AUDIT_LOG_FILE=/var/log/hed-bot/audit.log  # Log file location
```

### Log Retention

Recommended retention policies:

- **Audit logs**: 90 days minimum (compliance requirement)
- **Application logs**: 30 days
- **Access logs**: 30 days

**Logrotate Configuration**:
```bash
# /etc/logrotate.d/hed-bot
/var/log/hed-bot/*.log {
    daily
    rotate 90
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        docker kill -s USR1 hed-bot
    endscript
}
```

---

## Security Headers

All responses include security headers to protect against common attacks:

### Headers Implemented

```
X-Content-Type-Options: nosniff          # Prevent MIME type sniffing
X-Frame-Options: DENY                     # Prevent clickjacking
X-XSS-Protection: 1; mode=block           # Enable XSS filter
Strict-Transport-Security: max-age=31536000; includeSubDomains  # Force HTTPS
```

### Content Security Policy (CSP)

For API responses (added in Nginx):
```nginx
add_header Content-Security-Policy "default-src 'self'" always;
```

---

## Rate Limiting

### Nginx-Based Rate Limiting

**Configuration**:
```nginx
# In http block
limit_req_zone $binary_remote_addr zone=hed_bot_limit:10m rate=60r/m;

# In location block
limit_req zone=hed_bot_limit burst=10 nodelay;
limit_req_status 429;
```

**Limits**:
- **60 requests per minute** per IP address
- **10 request burst** allowed
- **429 status code** when exceeded

### Per-Endpoint Limits (Future)

Can be implemented with FastAPI slowapi:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/annotate")
@limiter.limit("10/minute")
async def annotate(...):
    ...
```

---

## HTTPS & Encryption

### Requirements

1. **HTTPS Only**: All production traffic must use HTTPS
2. **TLS 1.2+**: Minimum TLS version 1.2 (TLS 1.3 recommended)
3. **Strong Ciphers**: Use modern cipher suites

### Nginx TLS Configuration

```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256...';
ssl_prefer_server_ciphers on;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
```

### Certificate Management

- Use Let's Encrypt for free TLS certificates
- Auto-renewal via certbot
- HSTS header enforces HTTPS

---

## Environment Variables

### Secret Management

**Never commit secrets to Git!**

✅ **Correct**:
```bash
# .env file (gitignored)
OPENROUTER_API_KEY=sk-or-v1-abc123...
API_KEYS=key1,key2,key3
```

❌ **Incorrect**:
```python
# NEVER hardcode secrets in code
api_key = "sk-or-v1-abc123..."  # BAD!
```

### Environment File Template

```bash
# .env.example (committed to Git, no real values)
OPENROUTER_API_KEY=your_openrouter_key_here
API_KEYS=your_api_key_1,your_api_key_2
REQUIRE_API_AUTH=true
ENABLE_AUDIT_LOG=true
AUDIT_LOG_FILE=/var/log/hed-bot/audit.log
EXTRA_CORS_ORIGINS=
```

### File Permissions

```bash
# Restrict .env file permissions
chmod 600 .env
chown hed-bot:hed-bot .env

# Verify
ls -la .env
# Output: -rw------- 1 hed-bot hed-bot 256 Dec 02 15:30 .env
```

---

## Audit Compliance

### For Security Auditors

This section provides information for security auditors reviewing HED-BOT.

#### Security Controls Implemented

| Control | Implementation | Evidence |
|---------|---------------|----------|
| **Authentication** | API Key (64-char random) | `src/api/security.py` |
| **Authorization** | Endpoint-level auth required | `src/api/main.py` (Depends) |
| **Audit Logging** | All requests/responses logged | `src/api/security.py`, logs |
| **CORS** | Whitelist-based origin validation | `src/api/main.py` |
| **Encryption** | HTTPS/TLS 1.2+ required | Nginx config |
| **Input Validation** | Pydantic models | `src/api/models.py` |
| **Rate Limiting** | 60 req/min per IP | Nginx config |
| **Security Headers** | HSTS, X-Frame, CSP, etc. | Middleware |
| **Secret Management** | Environment variables only | `.env` (not in Git) |

#### Compliance Standards

**OWASP Top 10 (2021)**: Addressed

1. **A01:2021-Broken Access Control** ✅
   - API key authentication required
   - Audit logging tracks all access

2. **A02:2021-Cryptographic Failures** ✅
   - HTTPS only (enforced via HSTS)
   - Secrets in environment variables

3. **A03:2021-Injection** ✅
   - Input validation via Pydantic
   - Parameterized queries (no SQL injection)

4. **A04:2021-Insecure Design** ✅
   - Defense in depth (multiple layers)
   - Principle of least privilege

5. **A05:2021-Security Misconfiguration** ✅
   - Security headers configured
   - Debug mode disabled in production

6. **A06:2021-Vulnerable Components** ✅
   - Regular dependency updates
   - Automated security scanning (future)

7. **A07:2021-Authentication Failures** ✅
   - Strong API keys (64 characters)
   - Failed auth attempts logged

8. **A08:2021-Software & Data Integrity** ✅
   - Audit logs for all changes
   - Version control (Git)

9. **A09:2021-Logging & Monitoring** ✅
   - Comprehensive audit logs
   - Health checks and monitoring

10. **A10:2021-SSRF** ✅
    - No user-controlled URLs
    - LLM calls are to whitelisted APIs only

#### Audit Trail Access

**Viewing Audit Logs**:
```bash
# View recent audit logs
sudo tail -f /var/log/hed-bot/audit.log

# Search for specific API key usage
grep "api_key=a1b2c3d4" /var/log/hed-bot/audit.log

# View all requests from an IP
grep "ip=1.2.3.4" /var/log/hed-bot/audit.log

# View errors
grep "ERROR" /var/log/hed-bot/audit.log
```

#### Testing Security Controls

**Authentication Test**:
```bash
# Should fail (no API key)
curl https://hedtools.ucsd.edu/hed-bot-api/annotate

# Should succeed (valid API key)
curl -H "X-API-Key: valid_key" https://hedtools.ucsd.edu/hed-bot-api/annotate
```

**CORS Test**:
```bash
# Should include CORS headers
curl -H "Origin: https://hed-bot.pages.dev" \
     -I https://hedtools.ucsd.edu/hed-bot-api/health

# Should reject invalid origin
curl -H "Origin: https://evil.com" \
     -I https://hedtools.ucsd.edu/hed-bot-api/health
```

**Rate Limiting Test**:
```bash
# Rapid requests should trigger 429
for i in {1..70}; do
  curl https://hedtools.ucsd.edu/hed-bot-api/health
done
```

---

## Security Checklist

Use this checklist for deployment and audits:

### Pre-Deployment

- [ ] API keys generated and stored securely
- [ ] `.env` file has correct permissions (600)
- [ ] HTTPS certificate installed and valid
- [ ] Nginx configured with security headers
- [ ] Rate limiting enabled
- [ ] Audit logging enabled and tested
- [ ] CORS origins configured correctly
- [ ] Debug mode disabled (`DEBUG=false`)

### Post-Deployment

- [ ] Health check accessible: `https://hedtools.ucsd.edu/hed-bot-api/health`
- [ ] Authentication working (401 without API key)
- [ ] CORS headers present in responses
- [ ] Audit logs being written to `/var/log/hed-bot/audit.log`
- [ ] Security headers present in responses
- [ ] Rate limiting triggering after 60 req/min
- [ ] HTTPS redirect working (HTTP → HTTPS)

### Ongoing Maintenance

- [ ] Review audit logs weekly
- [ ] Rotate API keys quarterly
- [ ] Update dependencies monthly
- [ ] Review access logs for suspicious activity
- [ ] Test backup/restore procedures
- [ ] Update TLS certificates before expiry

---

## Incident Response

### If API Key is Compromised

1. **Immediately** remove compromised key from `.env`
2. Restart Docker container: `docker restart hed-bot`
3. Generate new API key: `python scripts/generate_api_key.py`
4. Update frontend with new key
5. Review audit logs for unauthorized access
6. Document incident for audit trail

### If Breach is Suspected

1. Check audit logs for suspicious activity
2. Review nginx access logs
3. Check for unauthorized API keys
4. Verify CORS origins haven't been modified
5. Review recent code changes
6. Contact security team

---

## Contact & Support

**Security Issues**: Report to security team immediately
**Audit Questions**: Contact project lead
**Implementation Questions**: See `deploy/README.md`

---

**Last Updated**: December 2, 2025
