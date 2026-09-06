import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class CrisisAssessment(BaseModel):
    is_crisis: bool = False
    hits: list[str] = Field(default_factory=list)
    source: str = "lexicon"


class FusionResult(BaseModel):
    composite_score: float
    confidence: float
    top_signals: list[str] = Field(default_factory=list)


class ForecastResult(BaseModel):
    p_escalation: float
    horizon_days: int = 7


class Decision(BaseModel):
    route: str  # crisis | escalate | routine
    reasons: list[str] = Field(default_factory=list)
    policy_version: str = ""


class OrchestratorState(TypedDict, total=False):
    thread_id: str
    case_id: str
    channel: str
    message: str
    language: str
    case_ctx: dict
    crisis: CrisisAssessment
    signals: Annotated[dict[str, Any], merge_dicts]
    fusion: FusionResult | None
    forecast: ForecastResult | None
    decision: Decision
    reply: str
    summary_text: str
    alert_id: str | None
    dispatched: bool
    counsellor_action: dict | None
    audit_ref: str
    errors: Annotated[list[str], operator.add]
    t_start_ms: float
    stream_q: Any  # queue.Queue when the caller wants SSE deltas
