from langgraph.graph import END, START, StateGraph

from src.orchestration import nodes_io as io
from src.orchestration import nodes_llm as ln
from src.orchestration.state import OrchestratorState as S


def build(cfg: dict, http, llm, checkpointer=None):
    g = StateGraph(S)

    def node(fn):
        """Bind cfg/http into a LangGraph node. LangGraph invokes node fns with
        (state) only — any additional dependencies must be closed over here."""
        return lambda s: fn(s, cfg=cfg, http=http)

    g.add_node("ingest", node(io.ingest))
    g.add_node("crisis_precheck", lambda s: io.crisis_precheck(s, cfg=cfg))
    g.add_node("fanout", lambda s: {})  # no-op fan-out point for parallel signals
    g.add_node("sentiment", node(io.make_signal_node("sentiment", "/v1/signals/text")))
    g.add_node("threat", node(io.make_signal_node("threat", "/v1/signals/threat")))
    g.add_node("voice", node(io.make_signal_node("voice", "/v1/signals/voice")))
    g.add_node("fuse", node(io.fuse))
    g.add_node("forecast", node(io.forecast))
    g.add_node("decide", lambda s: io.decide(s, cfg=cfg))
    g.add_node("victim_handoff", lambda s: io.victim_handoff(s, cfg=cfg))
    g.add_node("respond", lambda s: ln.respond(s, cfg=cfg, llm=llm))
    g.add_node("summarize", lambda s: ln.summarize(s, cfg=cfg, llm=llm))
    g.add_node("dispatch_crisis", node(io.dispatch_alert("crisis")))
    g.add_node("dispatch_escalation", node(io.dispatch_alert("high")))
    g.add_node("audit", lambda s: io.audit_node(s, cfg=cfg))

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "crisis_precheck")
    g.add_conditional_edges(
        "crisis_precheck",
        lambda s: "crisis" if s["crisis"].is_crisis else "fanout",
        {"crisis": "victim_handoff", "fanout": "fanout"},
    )
    g.add_edge("victim_handoff", "dispatch_crisis")
    g.add_edge("dispatch_crisis", "audit")
    g.add_edge("fanout", "sentiment")
    g.add_edge("fanout", "threat")
    g.add_edge("fanout", "voice")
    for n in ("sentiment", "threat", "voice"):
        g.add_edge(n, "fuse")
    g.add_edge("fuse", "forecast")
    g.add_edge("forecast", "decide")
    g.add_conditional_edges(
        "decide",
        lambda s: "summarize" if s["decision"].route == "escalate" else "respond",
        {"summarize": "summarize", "respond": "respond"},
    )
    g.add_edge("summarize", "dispatch_escalation")
    g.add_edge("dispatch_escalation", "audit")
    g.add_edge("respond", "audit")
    g.add_edge("audit", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=["dispatch_escalation"])
