import hashlib
import json
import pathlib
import threading


class AuditLedger:
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.prev_hash = "0" * 64
        if self.path.exists() and self.path.stat().st_size:
            self.prev_hash = json.loads(self.path.read_text().splitlines()[-1])["record_hash"]

    def append(self, record: dict) -> dict:
        with self._lock:
            record["previous_hash"] = self.prev_hash
            blob = json.dumps(record, sort_keys=True, separators=(",", ":"))
            record["record_hash"] = hashlib.sha256((self.prev_hash + blob).encode()).hexdigest()
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os_fsync(f)
            self.prev_hash = record["record_hash"]
            return record

    def verify(self) -> dict:
        """Recompute the full chain; returns {'valid': bool, 'n': int, 'last_hash': str}."""
        prev, n = "0" * 64, 0
        if not self.path.exists():
            return {"valid": True, "n": 0, "last_hash": prev}
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            blob = json.dumps({k: v for k, v in rec.items() if k != "record_hash"},
                              sort_keys=True, separators=(",", ":"))
            if rec.get("previous_hash") != prev or \
               hashlib.sha256((prev + blob).encode()).hexdigest() != rec.get("record_hash"):
                return {"valid": False, "n": n, "last_hash": prev}
            prev, n = rec["record_hash"], n + 1
        return {"valid": True, "n": n, "last_hash": prev}


def os_fsync(f):
    import os
    os.fsync(f.fileno())