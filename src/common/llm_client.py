import time
from pathlib import Path

import structlog
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.common.resilience import breakers

log = structlog.get_logger()
_TRANSIENT = (APIConnectionError, APITimeoutError, RateLimitError)
_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class LLMDown(Exception):
    pass


class LLMClient:
    SYSTEM_A = (_PROMPTS_DIR / "sahayak_system.txt").read_text(encoding="utf-8").strip()
    SYSTEM_B = (_PROMPTS_DIR / "casewriter_system.txt").read_text(encoding="utf-8").strip()

    def __init__(self, base_url: str, model_map: dict[str, str], api_key: str = "EMPTY", timeout: int = 25):
        self.cli = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.model_map = model_map
        self._breaker = breakers.get("llm")
        self._healthy_until = 0.0
        self._healthy = False

    # ---- internal -------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(_TRANSIENT),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5),
        reraise=True,
    )
    def _create(self, model: str, messages: list, temperature: float, max_tokens: int):
        return self.cli.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )

    def _model(self, adapter_key: str) -> str:
        try:
            return self.model_map[adapter_key]
        except KeyError as e:
            raise LLMDown(f"llm_model_missing:{adapter_key}") from e

    def _run(self, adapter_key: str, messages: list, temperature: float, max_tokens: int) -> str:
        model = self._model(adapter_key)
        if not self._breaker.allow():
            raise LLMDown("llm_breaker_open")
        try:
            r = self._create(model, messages, temperature, max_tokens)
        except _TRANSIENT as e:
            self._breaker.record(False)
            raise LLMDown(str(e)) from e
        self._breaker.record(True)
        return (r.choices[0].message.content or "").strip()

    # ---- public ---------------------------------------------------------
    def chat(self, adapter_key: str, messages: list, temperature: float = 0.6, max_tokens: int = 220) -> str:
        t0 = time.perf_counter()
        text = self._run(adapter_key, messages, temperature, max_tokens)
        log.info("llm_call", adapter=adapter_key, ms=int((time.perf_counter() - t0) * 1000))
        return text

    def stream(self, adapter_key: str, messages: list, temperature: float = 0.6, max_tokens: int = 220):
        """Generator of text chunks; breaker + transient-retry applied to the first call."""
        model = self._model(adapter_key)
        if not self._breaker.allow():
            raise LLMDown("llm_breaker_open")
        try:
            stream = self.cli.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True
            )
            for event in stream:
                if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
            self._breaker.record(True)
        except _TRANSIENT as e:
            self._breaker.record(False)
            raise LLMDown(str(e)) from e

    def health(self) -> bool:
        if time.monotonic() < self._healthy_until:
            return self._healthy
        try:
            self.cli.models.list()
            self._healthy, self._healthy_until = True, time.monotonic() + 15
        except Exception:
            self._healthy, self._healthy_until = False, time.monotonic() + 15
        return self._healthy
