"""
Google OAuth + JWT Session Authentication

Verifies Google ID tokens and issues short-lived JWTs for session management.
"""

import os
import logging
from typing import Optional
from datetime import datetime, timezone, timedelta

import jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

logger = logging.getLogger(__name__)

JWT_ALGORITHM = "HS256"


def _get_google_client_id() -> str:
    val = os.getenv("GOOGLE_CLIENT_ID", "")
    if not val:
        raise RuntimeError(
            "GOOGLE_CLIENT_ID environment variable is not set. "
            "Set it in .env or as an environment variable."
        )
    return val


def _get_jwt_secret() -> str:
    val = os.getenv("JWT_SECRET", "")
    if not val:
        raise RuntimeError(
            "JWT_SECRET environment variable is not set. "
            "Generate a random secret string and set it in .env or as an environment variable."
        )
    return val


def _get_jwt_expiry_hours() -> int:
    return int(os.getenv("JWT_EXPIRY_HOURS", "168"))  # 7 days default


def verify_google_token(token: str) -> Optional[dict]:
    """
    Verify a Google ID token and extract user info.

    Args:
        token: The Google ID token from frontend sign-in

    Returns:
        Dict with google_sub, email, name, avatar_url if valid; None if invalid
    """
    try:
        idinfo = id_token.verify_oauth2_token(
            token, google_requests.Request(), _get_google_client_id()
        )

        # Verify issuer
        if idinfo["iss"] not in ("accounts.google.com", "https://accounts.google.com"):
            logger.warning("Invalid issuer in Google token")
            return None

        return {
            "google_sub": idinfo["sub"],
            "email": idinfo.get("email", ""),
            "name": idinfo.get("name", ""),
            "avatar_url": idinfo.get("picture", ""),
        }
    except ValueError as e:
        logger.warning(f"Google token verification failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Google token verification: {e}")
        return None


def create_session_jwt(user_id: str, email: str, name: str) -> str:
    """
    Create a JWT for session authentication.

    Args:
        user_id: The internal user UUID
        email: User's email
        name: User's display name

    Returns:
        Encoded JWT string
    """
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=_get_jwt_expiry_hours()),
    }
    return jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def verify_session_jwt(token: str) -> Optional[dict]:
    """
    Verify a session JWT and extract user info.

    Returns:
        Dict with user_id, email, name if valid; None if invalid/expired
    """
    try:
        payload = jwt.decode(token, _get_jwt_secret(), algorithms=[JWT_ALGORITHM])
        return {
            "user_id": payload["sub"],
            "email": payload.get("email", ""),
            "name": payload.get("name", ""),
        }
    except jwt.ExpiredSignatureError:
        logger.debug("JWT expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug(f"JWT invalid: {e}")
        return None
