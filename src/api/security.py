"""Security middleware and utilities for HED-BOT API.

This module provides authentication, authorization, and audit logging
for API endpoints to ensure compliance with security best practices.
"""

import logging
import os
import secrets
from datetime import datetime

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

# Configure logging
logger = logging.getLogger(__name__)

# API Key header name
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Audit log format
AUDIT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [AUDIT] %(message)s"


class APIKeyAuth:
    """API Key authentication handler."""

    def __init__(self):
        """Initialize API key authentication."""
        # Set require_auth first (needed by _load_api_keys)
        self.require_auth = os.getenv("REQUIRE_API_AUTH", "true").lower() == "true"
        # Load API keys from environment
        self.api_keys = self._load_api_keys()

    def _load_api_keys(self) -> set[str]:
        """Load API keys from environment variables.

        Returns:
            Set of valid API key hashes
        """
        keys = set()

        # Load from comma-separated API_KEYS env var
        if api_keys_str := os.getenv("API_KEYS"):
            keys.update(key.strip() for key in api_keys_str.split(",") if key.strip())

        # Load individual keys (API_KEY_1, API_KEY_2, etc.)
        i = 1
        while api_key := os.getenv(f"API_KEY_{i}"):
            keys.add(api_key.strip())
            i += 1

        # If no keys configured, generate a warning
        if not keys and self.require_auth:
            logger.warning(
                "No API keys configured! Set API_KEYS environment variable. "
                "Authentication is enabled but no keys are set."
            )

        return keys

    def verify_api_key(self, api_key: str | None) -> bool:
        """Verify if an API key is valid.

        Args:
            api_key: API key to verify

        Returns:
            True if key is valid, False otherwise
        """
        if not self.require_auth:
            return True

        if not api_key:
            return False

        # Check if key exists in configured keys
        return api_key in self.api_keys

    async def __call__(self, api_key: str | None = Security(API_KEY_HEADER)) -> str:
        """FastAPI dependency for API key authentication.

        Args:
            api_key: API key from request header

        Returns:
            The validated API key

        Raises:
            HTTPException: If authentication fails
        """
        # If authentication is disabled, allow all requests
        if not self.require_auth:
            return "auth_disabled"

        # If no API key provided
        if not api_key:
            logger.warning("Request rejected: No API key provided")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required. Include X-API-Key header.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # Verify API key
        if not self.verify_api_key(api_key):
            # Log only first 8 chars for debugging without revealing full key
            # Do not log any part of the API key for security reasons
            logger.warning("Request rejected: Invalid API key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # Do not log any part of the API key for security reasons
        logger.info("Request authenticated with API key")
        return api_key


class AuditLogger:
    """Audit logging for API requests and responses."""

    def __init__(self):
        """Initialize audit logger."""
        self.logger = logging.getLogger("hed_bot.audit")
        self.enabled = os.getenv("ENABLE_AUDIT_LOG", "true").lower() == "true"

        # Configure audit log file if enabled
        if self.enabled:
            log_file = os.getenv("AUDIT_LOG_FILE", "/var/log/hed-bot/audit.log")
            try:
                handler = logging.FileHandler(log_file)
                handler.setFormatter(logging.Formatter(AUDIT_LOG_FORMAT))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
            except (PermissionError, FileNotFoundError) as e:
                logger.warning(f"Could not configure audit log file: {e}")
                # Fall back to console logging
                self.logger.addHandler(logging.StreamHandler())

    def log_request(
        self,
        request: Request,
        api_key_hash: str | None = None,
        user_id: str | None = None,
    ):
        """Log an API request for audit purposes.

        Args:
            request: FastAPI request object
            api_key_hash: Hash of API key used (first 8 chars)
            user_id: Optional user identifier
        """
        if not self.enabled:
            return

        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        timestamp = datetime.utcnow().isoformat()

        self.logger.info(
            f"REQUEST - "
            f"timestamp={timestamp}, "
            f"ip={client_ip}, "
            f"method={method}, "
            f"path={path}, "
            f"api_key={api_key_hash or 'none'}, "
            f"user={user_id or 'anonymous'}"
        )

    def log_response(
        self,
        request: Request,
        status_code: int,
        processing_time_ms: float,
    ):
        """Log an API response for audit purposes.

        Args:
            request: FastAPI request object
            status_code: HTTP status code
            processing_time_ms: Request processing time in milliseconds
        """
        if not self.enabled:
            return

        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        timestamp = datetime.utcnow().isoformat()

        self.logger.info(
            f"RESPONSE - "
            f"timestamp={timestamp}, "
            f"ip={client_ip}, "
            f"method={method}, "
            f"path={path}, "
            f"status={status_code}, "
            f"duration_ms={processing_time_ms:.2f}"
        )

    def log(
        self,
        event: str,
        data: dict | None = None,
    ):
        """Log a general event for audit purposes.

        Args:
            event: Event name/type
            data: Optional event data dictionary
        """
        if not self.enabled:
            return

        timestamp = datetime.utcnow().isoformat()
        data_str = ", ".join(f"{k}={v}" for k, v in (data or {}).items())

        self.logger.info(f"EVENT - timestamp={timestamp}, event={event}, {data_str}")

    def log_error(
        self,
        request: Request,
        error: Exception,
        api_key_hash: str | None = None,
    ):
        """Log an error for audit purposes.

        Args:
            request: FastAPI request object
            error: Exception that occurred
            api_key_hash: Hash of API key used
        """
        if not self.enabled:
            return

        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        timestamp = datetime.utcnow().isoformat()
        error_type = type(error).__name__
        error_msg = str(error)

        self.logger.error(
            f"ERROR - "
            f"timestamp={timestamp}, "
            f"ip={client_ip}, "
            f"method={method}, "
            f"path={path}, "
            f"api_key={api_key_hash or 'none'}, "
            f"error_type={error_type}, "
            f"error_msg={error_msg}"
        )


# Global instances
api_key_auth = APIKeyAuth()
audit_logger = AuditLogger()


def generate_api_key() -> str:
    """Generate a secure random API key.

    Returns:
        A 32-character hexadecimal API key
    """
    return secrets.token_hex(32)


def verify_origin(request: Request, allowed_origins: list[str]) -> bool:
    """Verify request origin is in allowed list.

    Args:
        request: FastAPI request object
        allowed_origins: List of allowed origin URLs

    Returns:
        True if origin is allowed, False otherwise
    """
    origin = request.headers.get("origin") or request.headers.get("referer")

    if not origin:
        # No origin header (direct API access) - allow if API key is valid
        return True

    # Check if origin matches any allowed origins
    for allowed in allowed_origins:
        if origin.startswith(allowed):
            return True

    return False
