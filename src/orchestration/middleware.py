import contextvars
import threading
import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.common.settings import settings

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
log = structlog.get_logger()


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        request_id_ctx.set(rid)
        structlog.contextvars.bind_contextvars(request_id=rid)
        t0 = time.perf_counter()
        resp = await call_next(request)
        resp.headers["X-Request-ID"] = rid
        log.info(
            "access",
            method=request.method,
            path=request.url.path,
            status=resp.status_code,
            ms=int((time.perf_counter() - t0) * 1000),
        )
        return resp


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        resp.headers.update(
            {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer"}
        )
        return resp


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket per API key (per worker). Swap the store for Redis to scale out."""

    def __init__(self, app):
        super().__init__(app)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    async def dispatch(self, request, call_next):
        key = request.headers.get("X-API-Key") or (request.client.host if request.client else "anon")
        cap, refill = settings.rate_limit_rpm, settings.rate_limit_rpm / 60.0
        now = time.monotonic()
        with self._lock:
            tokens, ts = self._buckets.get(key, (float(cap), now))
            tokens = min(cap, tokens + (now - ts) * refill)
            if tokens < 1:
                return JSONResponse({"error": "rate_limited"}, status_code=429, headers={"Retry-After": "5"})
            self._buckets[key] = (tokens - 1, now)
        return await call_next(request)
