import json

from src.orchestration.audit import AuditLedger

KEY = b"k" * 32


def test_roundtrip_and_verify(tmp_path):
    led = AuditLedger(str(tmp_path / "l.jsonl"), KEY)
    for i in range(5):
        led.append({"i": i, "data": "x" * 100})
    v = led.verify()
    assert v["valid"] and v["n"] == 5


def test_tamper_detected(tmp_path):
    p = tmp_path / "l.jsonl"
    led = AuditLedger(str(p), KEY)
    led.append({"i": 0})
    led.append({"i": 1})
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["i"] = 999
    lines[1] = json.dumps(rec)
    p.write_text("\n".join(lines))
    assert AuditLedger(str(p), KEY).verify()["valid"] is False


def test_forgery_resistant_with_wrong_key(tmp_path):
    p = tmp_path / "l.jsonl"
    AuditLedger(str(p), KEY).append({"i": 0})
    attacker = AuditLedger(str(p), b"z" * 32)
    assert attacker.verify()["valid"] is False


def test_rotation_continues_chain(tmp_path):
    p = tmp_path / "l.jsonl"
    led = AuditLedger(str(p), KEY)
    led.append({"i": 0})
    p.rename(p.with_suffix(".1.jsonl"))
    led2 = AuditLedger(str(p), KEY)
    led2.append({"i": 1})
    assert led2.verify()["valid"]
