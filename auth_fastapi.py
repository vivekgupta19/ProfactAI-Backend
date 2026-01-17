from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Header, HTTPException

from database import db


def get_current_user_from_auth_header(
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """
    FastAPI dependency that validates a Bearer token and returns the current user
    based on the active session record in the database.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    session = db.get_session_by_token(token)
    if not session:
        raise HTTPException(status_code=401, detail="Authentication required")

    return {
        "userId": session["userid"],
        "email": session["email"],
        "fullName": session["fullname"],
        "role": session["role"],
        "organizationId": session["organizationid"],
        "organizationName": session["organizationname"],
        "organizationSlug": session["organizationslug"],
        "_token": token,  # internal convenience for logout
    }

