import os
import tempfile

# Set env BEFORE any src import — Settings reads env at import time
_tmpdir = tempfile.mkdtemp()
os.environ.setdefault("ORCH_ENV", "dev")
os.environ.setdefault("ORCH_API_KEYS", '{"victim":"vk_t","counsellor":"ck_t","ops":"ok_t"}')
os.environ.setdefault("ORCH_LLM_BASE_URL", "http://127.0.0.1:9")  # deliberately dead LLM
os.environ.setdefault("ORCH_AUDIT_PATH", os.path.join(_tmpdir, "test_ledger.jsonl"))

import pytest  # noqa: E402

from src.common.resilience import breakers  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_circuit_breakers():
    breakers.reset()
    yield
    breakers.reset()
