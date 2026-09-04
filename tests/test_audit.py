import json

from src.orchestration.audit import AuditLedger


def test_append_and_verify(tmp_path):
    led = AuditLedger(str(tmp_path / "l.jsonl"))
    for i in range(3):
        led.append({"i": i})
    assert led.verify() == {"valid": True, "n": 3, "last_hash": led.prev_hash}


def test_tamper_detection(tmp_path):
    p = tmp_path / "l.jsonl"
    led = AuditLedger(str(p))
    led.append({"i": 0})
    led.append({"i": 1})
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["i"] = 999  # tamper
    lines[1] = json.dumps(rec)
    p.write_text("\n".join(lines))
    led2 = AuditLedger(str(p))
    assert led2.verify()["valid"] is False
