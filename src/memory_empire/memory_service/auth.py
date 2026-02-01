from __future__ import annotations

from fastapi import Header, HTTPException

from .settings import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.api_key:
        return
    if (x_api_key or "") != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid API key")
