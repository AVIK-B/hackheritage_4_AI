import hmac
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from src.common.settings import settings

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require(scope: str):
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