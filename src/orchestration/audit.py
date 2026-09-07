import hashlib
import hmac
import json
import os
import pathlib
import threading

MAX_FILE_BYTES = 32 * 1024 * 1024


class AuditLedger:
    """HMAC-SHA256 chained ledger. Without the key, entries cannot be forged
    or silently altered (tamper-resistant, not merely tamper-evident)."""

    def __init__(self, path: str, key: bytes):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._key = key
        self._lock = threading.Lock()
        self.prev_hash = "0" * 64
        if self.path.exists() and self.path.stat().st_size:
            self.prev_hash = json.loads(self.path.read_text(encoding="utf-8").splitlines()[-1])["record_hash"]

    def _mac(self, prev: str, blob: str) -> str:
        return hmac.new(self._key, (prev + blob).encode(), hashlib.sha256).hexdigest()

    def append(self, record: dict) -> dict:
        with self._lock:
            if self.path.exists() and self.path.stat().st_size > MAX_FILE_BYTES:
                self.path.rename(self.path.with_suffix(".1.jsonl"))
            record["previous_hash"] = self.prev_hash
            blob = json.dumps(record, sort_keys=True, separators=(",", ":"))
            record["record_hash"] = self._mac(self.prev_hash, blob)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.prev_hash = record["record_hash"]
            return record

    def verify(self) -> dict:
        prev, n = "0" * 64, 0
        if not self.path.exists():
            return {"valid": True, "n": 0, "last_hash": prev}
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                blob = json.dumps(
                    {k: v for k, v in rec.items() if k != "record_hash"},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if rec.get("previous_hash") != prev or not hmac.compare_digest(
                    self._mac(prev, blob), rec.get("record_hash", "")
                ):
                    return {"valid": False, "n": n, "last_hash": prev}
                prev, n = rec["record_hash"], n + 1
        return {"valid": True, "n": n, "last_hash": prev}
