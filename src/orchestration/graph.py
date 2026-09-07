from langgraph.graph import END, START, StateGraph

from src.common.safety import CRISIS_COPY
from src.orchestration import nodes_llm as ln
from src.orchestration.agents import (
    AgentRegistry,
    ContextAgent,
    CrisisPrecheckAgent,
    ForecastAgent,
    FusionAgent,
    SentimentAgent,
    ThreatAgent,
    VoiceAgent,
)
from src.orchestration.policies import evaluate
from src.orchestration.state import Decision, ForecastResult, FusionResult
from src.orchestration.state import OrchestratorState as S


def build(cfg: dict, http, llm, checkpointer=None):
    registry = AgentRegistry(cfg, http, llm)

    def agent_node(cls):
        agent = registry.get(cls)

        def _node(state: S, config=None) -> dict:
            from src.orchestration.agents import Deadline

            deadline = Deadline(budget_ms=12000)  # or read from config
            state = {**state, "_deadline": deadline}
            res = agent.run(state)
            ...
            upd: dict = {
                "_agent_telemetry": {
                    **(state.get("_agent_telemetry") or {}),
                    res.agent: {"ok": res.ok, "degraded": res.degraded, "ms": res.latency_ms},
                }
            }
            if cls is CrisisPrecheckAgent:
                upd["crisis"] = res.data
            elif cls is ContextAgent:
                upd["case_ctx"] = res.data
            elif cls is SentimentAgent:
                upd["signals"] = {**(state.get("signals") or {}), "sentiment": res.data}
            elif cls is ThreatAgent:
                upd["signals"] = {**(state.get("signals") or {}), "threat": res.data}
            elif cls is VoiceAgent and res.data is not None:
                upd["signals"] = {**(state.get("signals") or {}), "voice": res.data}
            elif cls is FusionAgent and res.data is not None:
                upd["fusion"] = FusionResult(**res.data)
            elif cls is ForecastAgent and res.data is not None:
                upd["forecast"] = ForecastResult(**res.data)
            return upd

        return _node

    def parallel_signals(state: S) -> dict:
        res = registry.parallel([SentimentAgent, ThreatAgent, ContextAgent], state)
        upd = {
            "_agent_telemetry": {
                **(state.get("_agent_telemetry") or {}),
                **{n: {"ok": r.ok, "degraded": r.degraded, "ms": r.latency_ms} for n, r in res.items()},
            }
        }
        sig = dict(state.get("signals") or {})
        if res["sentiment"].data:
            sig["sentiment"] = res["sentiment"].data
        if res["threat"].data:
            sig["threat"] = res["threat"].data
        upd["signals"] = sig
        if res["case_context"].data:
            upd["case_ctx"] = res["case_context"].data
            upd["language"] = res["case_context"].data.get("language", state.get("language", "en"))
        return upd

    def decide(state: S) -> dict:
        d = evaluate(
            state.get("crisis"),
            state.get("fusion"),
            state.get("forecast"),
            {k: v for k, v in (state.get("signals") or {}).items() if v},
            cfg,
        )
        return {"decision": d}

    def victim_handoff(state: S) -> dict:
        c = state["crisis"]
        hits = ", ".join(c.hits) if c and c.hits else "pattern match"
        return {
            "reply": CRISIS_COPY.get(state.get("language", "en"), CRISIS_COPY["en"]),
            "decision": Decision(
                route="crisis",
                reasons=[f"crisis precheck: {c.source} ({hits})"],
                policy_version=cfg.get("policy_version", "?"),
            ),
        }

    g = StateGraph(S)
    g.add_node("crisis_precheck", agent_node(CrisisPrecheckAgent))
    g.add_node("parallel_signals", parallel_signals)
    g.add_node("fuse", agent_node(FusionAgent))
    g.add_node("forecast", agent_node(ForecastAgent))
    g.add_node("decide", decide)
    g.add_node("victim_handoff", victim_handoff)
    g.add_node("respond", lambda s: ln.respond(s, cfg=cfg, llm=llm, registry=None))
    g.add_node("summarize", lambda s: ln.summarize(s, cfg=cfg, llm=llm, registry=None))
    g.add_node("dispatch", lambda s: ln.dispatch(s, cfg=cfg, llm=llm, registry=None))
    g.add_node("audit", lambda s: s.get("_audit_passthrough", {}))

    g.add_edge(START, "crisis_precheck")
    g.add_conditional_edges(
        "crisis_precheck",
        lambda s: "crisis" if s["crisis"].is_crisis else "parallel_signals",
        {"crisis": "victim_handoff", "parallel_signals": "parallel_signals"},
    )
    g.add_edge("victim_handoff", "dispatch")
    g.add_edge("parallel_signals", "fuse")
    g.add_edge("fuse", "forecast")
    g.add_edge("forecast", "decide")
    g.add_conditional_edges(
        "decide",
        lambda s: "summarize" if s["decision"].route == "escalate" else "respond",
        {"summarize": "summarize", "respond": "respond"},
    )
    g.add_edge("summarize", "dispatch")
    g.add_edge("dispatch", "audit")
    g.add_edge("respond", "audit")
    g.add_edge("audit", END)

    app = g.compile(checkpointer=checkpointer, interrupt_before=["dispatch"])
    app._registry = registry
    return app
