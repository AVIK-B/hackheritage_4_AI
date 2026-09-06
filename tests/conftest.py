import os

import pytest

from src.common.resilience import breakers

os.environ.setdefault("ORCH_ENV", "dev")
os.environ.setdefault("ORCH_API_KEYS", '{"victim":"vk_t","counsellor":"ck_t","ops":"ok_t"}')
os.environ.setdefault("ORCH_LLM_BASE_URL", "http://127.0.0.1:9")  # deliberately dead LLM
os.environ.setdefault("ORCH_AUDIT_PATH", "./data/test_ledger.jsonl")


@pytest.fixture(autouse=True)
def _isolated_circuit_breakers():
    breakers.reset()
    yield
    breakers.reset()
