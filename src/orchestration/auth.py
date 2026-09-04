import hmac

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from src.common.settings import settings

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require(scope: str):
    """Dependency factory: returns an async FastAPI dependency for the given scope.
    NOTE: `require` itself must stay sync — an async factory called with ()
    returns a coroutine object instead of the dependency, which breaks FastAPI."""

    async def dep(x_api_key: str | None = Security(_header)) -> str:
        if not x_api_key:
            raise HTTPException(401, "missing X-API-Key")
        for role, key in settings.api_keys.items():
            if hmac.compare_digest(x_api_key, key):
                if scope == role or role == "ops":
                    return role
                raise HTTPException(403, f"key lacks '{scope}' scope")
        raise HTTPException(401, "invalid key")

    return dep
