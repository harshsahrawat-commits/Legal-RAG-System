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

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "168"))  # 7 days default


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
            token, google_requests.Request(), GOOGLE_CLIENT_ID
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
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_session_jwt(token: str) -> Optional[dict]:
    """
    Verify a session JWT and extract user info.

    Returns:
        Dict with user_id, email, name if valid; None if invalid/expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
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
