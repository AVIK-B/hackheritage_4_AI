import os

os.environ.setdefault("ORCH_ENV", "dev")
os.environ.setdefault("ORCH_API_KEYS", '{"victim":"vk_t","counsellor":"ck_t","ops":"ok_t"}')
os.environ.setdefault("ORCH_LLM_BASE_URL", "http://127.0.0.1:9")  # deliberately dead LLM
os.environ.setdefault("ORCH_AUDIT_PATH", "./data/test_ledger.jsonl")
