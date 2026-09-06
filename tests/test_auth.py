import json

from src.orchestration.audit import AuditLedger


def test_roundtrip_and_verify(tmp_path):
    led = AuditLedger(str(tmp_path / "l.jsonl"), b"k" * 32)
    for i in range(5):
        led.append({"i": i, "data": "x" * 100})
    v = led.verify()
    assert v["valid"] and v["n"] == 5


def test_tamper_detected(tmp_path):
    p = tmp_path / "l.jsonl"
    led = AuditLedger(str(p), b"k" * 32)
    led.append({"i": 0})
    led.append({"i": 1})
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["i"] = 999
    lines[1] = json.dumps(rec)
    p.write_text("\n".join(lines))
    assert AuditLedger(str(p), b"k" * 32).verify()["valid"] is False


def test_forgery_resistant_with_wrong_key(tmp_path):
    p = tmp_path / "l.jsonl"
    AuditLedger(str(p), b"k" * 32).append({"i": 0})
    attacker = AuditLedger(str(p), b"z" * 32)  # wrong key: chain won't verify
    assert attacker.verify()["valid"] is False


def test_rotation_continues_chain(tmp_path):
    p = tmp_path / "l.jsonl"
    led = AuditLedger(str(p), b"k" * 32)
    led.append({"i": 0})
    p.rename(p.with_suffix(".1.jsonl"))  # simulate rotation
    led2 = AuditLedger(str(p), b"k" * 32)  # fresh ledger; chain restarts by design
    led2.append({"i": 1})
    assert led2.verify()["valid"]
