import threading
import time

import structlog

log = structlog.get_logger()


class CircuitBreaker:
    """Closed -> (failures>=threshold) -> OPEN for reset_timeout -> HALF-OPEN (single probe)."""

    def __init__(self, name: str, failure_threshold: int = 3, reset_timeout: float = 30.0):
        self.name, self.threshold, self.reset_timeout = name, failure_threshold, reset_timeout
        self._failures, self._opened_at, self._lock = 0, None, threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if time.monotonic() - self._opened_at >= self.reset_timeout:
                return "half_open"
            return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._failures, self._opened_at = 0, None
            else:
                self._failures += 1
                if self._failures >= self.threshold:
                    self._opened_at = time.monotonic()
                    log.warning("breaker_open", breaker=self.name)


class Breakers:
    def __init__(self) -> None:
        self._map: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            if name not in self._map:
                self._map[name] = CircuitBreaker(name)
            return self._map[name]

    def reset(self) -> None:
        with self._lock:
            self._map.clear()


breakers = Breakers()
