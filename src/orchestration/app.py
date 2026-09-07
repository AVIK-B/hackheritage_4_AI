import json
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import psycopg_pool
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.common.llm_client import LLMClient
from src.common.logging import configure_logging
from src.common.settings import settings
from src.orchestration import auth
from src.orchestration.audit import AuditLedger
from src.orchestration.graph import build
from src.orchestration.middleware import (
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)

configure_logging()
log = structlog.get_logger()
INTERACTIONS = Counter("orch_interactions_total", "interactions", ["route", "channel"])
GRAPH_LAT = Histogram("orch_graph_latency_seconds", "graph latency", buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20))
_state: dict = {}


class Interaction(BaseModel):
    case_id: str = Field(min_length=4, max_length=64)
    channel: str = Field(pattern="^(pwa|sms|ivr|email)$")
    message: str = Field(min_length=1, max_length=4000)
    language: str | None = Field(default=None, max_length=8)


class ResumeAction(BaseModel):
    approved: bool
    notes: str = Field(default="", max_length=2000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["ledger"] = AuditLedger(settings.audit_path, settings.audit_hmac_key.encode())
    _state["http"] = httpx.Client()
    _state["llm"] = LLMClient(
        settings.llm_base_url, settings.llm_model_map, settings.llm_api_key, settings.llm_timeout_s
    )
    if settings.env == "dev" or not settings.checkpoint_dsn:
        _state["cp"] = MemorySaver()
        log.warning("memory_checkpointer", reason="dev_only")
    else:
        pool = psycopg_pool.ConnectionPool(settings.checkpoint_dsn, min_size=2, max_size=6)
        cp = PostgresSaver(pool)
        cp.setup()
        _state["cp"] = cp
    _state["graph"] = build(settings.model_dump(), _state["http"], _state["llm"], _state["cp"])
    _state["gate"] = threading.BoundedSemaphore(settings.max_concurrent_graphs)
    log.info("orchestrator_up", env=settings.env, llm=settings.llm_base_url)
    yield
    _state["http"].close()
    log.info("orchestrator_down")


app = FastAPI(
    title="PS-26094 Orchestrator API",
    version="2.0.0",
    docs_url="/docs" if settings.env != "prod" else None,
    lifespan=lifespan,
)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.exception_handler(HTTPException)
async def http_err(request: Request, exc: HTTPException):
    return JSONResponse({"error": "request_rejected", "detail": exc.detail}, exc.status_code)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled_error", path=request.url.path)
    return JSONResponse({"error": "internal_error"}, 500)


def _invoke(init: dict, stream_q: queue.Queue | None = None) -> dict:
    init.setdefault("thread_id", uuid.uuid4().hex)
    cfg = {"configurable": {"thread_id": init["thread_id"]}}
    if stream_q is not None:
        init["stream_q"] = stream_q
    with _state["gate"], GRAPH_LAT.time():
        final = _state["graph"].invoke(init, cfg)
    ...
    INTERACTIONS.labels(final["decision"].route, init["channel"]).inc()
    final["thread_id"] = init["thread_id"]

    # tamper-evident ledger write (outside graph state: JSONL + hash chain)
    # tamper-evident ledger write
    rec = _state["ledger"].append(
        {
            "kind": "interaction",
            "thread_id": init["thread_id"],
            "case_id": init["case_id"],
            "channel": init["channel"],
            "decision": final["decision"].model_dump(),
            "fusion": final["fusion"].model_dump() if final.get("fusion") else None,
            "forecast": final["forecast"].model_dump() if final.get("forecast") else None,
            "errors": final.get("errors", []),
            "alert_id": final.get("alert_id"),
            "telemetry": final.get("_agent_telemetry", {}),
        }
    )
    final["audit_ref"] = rec["record_hash"]  # ← THIS LINE WAS MISSING
    return final


@app.post("/v1/interactions")
def interact(req: Interaction, role: str = Depends(auth.require("victim"))):
    t0 = time.perf_counter()
    final = _invoke(
        {
            "case_id": req.case_id,
            "channel": req.channel,
            "message": req.message,
            "language": req.language or "en",
            "signals": {},
            "errors": [],
            "dispatched": False,
        }
    )
    awaiting = final["decision"].route == "escalate" and not final.get("dispatched")
    log.info("interaction_done", route=final["decision"].route, ms=int((time.perf_counter() - t0) * 1000))
    return {
        "thread_id": final["thread_id"],
        "reply": final.get("reply", ""),
        "status": "awaiting_counsellor" if awaiting else "completed",
        "audit_ref": final.get("audit_ref"),
    }


@app.post("/v1/interactions/stream")
def interact_stream(req: Interaction, role: str = Depends(auth.require("victim"))):
    q: queue.Queue = queue.Queue()
    result: dict = {}

    def run():
        try:
            result["final"] = _invoke(
                {
                    "case_id": req.case_id,
                    "channel": req.channel,
                    "message": req.message,
                    "language": req.language or "en",
                    "signals": {},
                    "errors": [],
                    "dispatched": False,
                },
                stream_q=q,
            )
        except Exception as e:
            result["err"] = str(e)
        q.put(None)

    threading.Thread(target=run, daemon=True).start()

    def gen():
        while True:
            chunk = q.get()
            if chunk is None:
                break
            yield f"data: {json.dumps({'delta': chunk})}\n\n"
        final = result.get("final")
        # replace the yield block in gen() of interact_stream:
        if final:
            awaiting = final["decision"].route == "escalate" and not final.get("dispatched")
            payload = {
                "thread_id": final["thread_id"],
                "status": "awaiting_counsellor" if awaiting else "completed",
                "audit_ref": final.get("audit_ref"),
                "reply": final.get("reply", ""),
            }
            yield "data: " + json.dumps({"final": payload}) + "\n\n"
        else:
            yield "data: " + json.dumps({"error": result.get("err", "internal_error")}) + "\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/v1/counsellor/{thread_id}/resume")
def resume(thread_id: str, action: ResumeAction, role: str = Depends(auth.require("counsellor"))):
    cfg = {"configurable": {"thread_id": thread_id}}
    _state["graph"].update_state(cfg, {"counsellor_action": action.model_dump()})
    final = _state["graph"].invoke(None, cfg)
    _state["ledger"].append(
        {
            "kind": "counsellor_resume",
            "thread_id": thread_id,
            "approved": action.approved,
            "notes": action.notes,
            "alert_id": final.get("alert_id"),
        }
    )
    return {"status": "completed", "alert_id": final.get("alert_id"), "audit_ref": final.get("audit_ref")}


@app.get("/v1/threads/{thread_id}/trace")
def trace(thread_id: str, role: str = Depends(auth.require("counsellor"))):
    snap = _state["graph"].get_state({"configurable": {"thread_id": thread_id}})
    v = snap.values or {}
    return {
        "decision": v.get("decision"),
        "fusion": v.get("fusion"),
        "forecast": v.get("forecast"),
        "signals": list(v.get("signals", {})),
        "errors": v.get("errors", []),
        "telemetry": v.get("_agent_telemetry", {}),
        "next": snap.next,
    }


@app.get("/v1/audit/verify")
def audit_verify(role: str = Depends(auth.require("ops"))):
    return _state["ledger"].verify()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz(role: str = Depends(auth.require("ops"))):
    reg = getattr(_state["graph"], "_registry", None)
    return {
        "llm": _state["llm"].health(),
        "graph": _state["graph"] is not None,
        "agents": reg.health_report() if reg else {},
        "degradation_level": reg.degradation_level() if reg else 0,
    }


@app.get("/metrics")
def metrics(role: str = Depends(auth.require("ops"))):
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
