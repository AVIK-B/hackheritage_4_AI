from src.orchestration.nodes_io import dispatch_alert, init_ledger
from src.orchestration.state import Decision


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"alert_id": "ALERT-1"}


class FakeHTTP:
    def post(self, url, json, timeout):
        self.payload = json
        return FakeResponse()


def test_dispatch_alert_does_not_mutate_state_with_internal_severity(tmp_path):
    init_ledger(str(tmp_path / "ledger.jsonl"))
    http = FakeHTTP()
    state = {
        "thread_id": "t1",
        "case_id": "CASE-1",
        "decision": Decision(route="crisis", reasons=["precheck"]),
    }
    cfg = {"services": {"backend": "http://backend"}}

    out = dispatch_alert("crisis")(state, cfg=cfg, http=http)

    assert out == {"dispatched": True, "alert_id": "ALERT-1"}
    assert http.payload["severity"] == "crisis"
    assert "_severity" not in state
