# PS-26094 Orchestrator

Safety-first orchestration API for victim support interactions. The service accepts
messages from PWA/SMS/IVR/email channels, enriches them with case context and risk
signals, makes an explainable routing decision, and records every interaction in a
tamper-evident audit ledger.

## Architecture

The runtime is a FastAPI app backed by a LangGraph workflow:

1. `ingest` fetches case context from the backend and scrubs contact PII.
2. `crisis_precheck` applies a local lexicon before any model call.
3. `sentiment`, `threat`, and `voice` call scoring services in parallel.
4. `fuse` combines signals into a composite risk score.
5. `forecast` estimates escalation probability.
6. `decide` applies versioned policy thresholds.
7. `respond` uses the dialogue model for routine replies.
8. `summarize` uses the casewriter model for counsellor-facing escalation notes.
9. `dispatch_*` calls the backend alerts endpoint.
10. `audit` appends the final state to the hash-chained ledger.

Crisis messages short-circuit directly to templated victim handoff copy and alert
dispatch. The LLM is never required for crisis handling.

## Main Components

- API: `src/orchestration/app.py`
- Graph wiring: `src/orchestration/graph.py`
- I/O and service nodes: `src/orchestration/nodes_io.py`
- LLM nodes: `src/orchestration/nodes_llm.py`
- Policy logic: `src/orchestration/policies.py`
- Audit ledger: `src/orchestration/audit.py`
- Settings: `src/common/settings.py`
- Prompts: `prompts/sahayak_system.txt`, `prompts/casewriter_system.txt`
- Training: `src/training/train_sft.py`
- GGUF export: `src/deployment/export_gguf.py`

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-orch.txt -r requirements-dev.txt
Copy-Item .env.example .env
```

Start the API:

```powershell
uvicorn src.orchestration.app:app --port 8500
```

Run checks:

```powershell
ruff format src tests scripts
ruff check src tests
python -m pytest -q
python scripts/verify_ledger.py
```

On Windows, if pytest cannot access the default temp/cache directories, use:

```powershell
python -m pytest -q --basetemp .tmp\pytest -o cache_dir=.tmp\pytest-cache
```

## Smoke And Demo

With the API running on port `8500`:

```bash
BASE=http://localhost:8500 bash scripts/smoke.sh
python scripts/demo_trace.py
```

The service tolerates missing downstream services in dev mode. Unavailable scoring,
fusion, forecast, backend, or LLM calls are recorded in `errors` and degrade to safe
fallback behavior.

## Configuration

Runtime configuration is read from environment variables with the `ORCH_` prefix.
Important values:

- `ORCH_ENV`: `dev` or `prod`
- `ORCH_API_KEYS`: JSON map for `victim`, `counsellor`, and `ops`
- `ORCH_LLM_BASE_URL`: OpenAI-compatible endpoint, usually Ollama or vLLM
- `ORCH_LLM_MODEL_MAP`: JSON map for `dialogue` and `summary`
- `ORCH_SERVICES`: JSON map for `scoring`, `fusion`, `forecast`, and `backend`
- `ORCH_CHECKPOINT_DSN`: optional Postgres checkpoint DSN
- `ORCH_AUDIT_PATH`: ledger path

Reference YAML files in `configs/` document expected dev, Azure, and Kaggle values.

## Models

The app expects two OpenAI-compatible chat models:

- `sahayak`: victim-facing dialogue adapter
- `casewriter`: counsellor-facing grounded summary adapter

Local Ollama models can be created from the files in `models/` after the GGUF files
are available:

```bash
ollama create sahayak -f models/Modelfile.sahayak
ollama create casewriter -f models/Modelfile.casewriter
```

## Deployment

CPU compose:

```bash
docker compose -f docker-compose.cpu.yml up -d --build
```

Production compose:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

Both compose files expose the API on `8500` and keep audit data on a named volume.
Production mode expects non-dev API keys and a real Postgres checkpoint backend.

## Security Notes

- Do not commit live tokens or production API keys.
- Rotate any token that has been committed or shared.
- Keep `ORCH_ENV=prod` deployments behind TLS and trusted ingress.
- The audit ledger is append-only at the application level; use durable storage for
  `ORCH_AUDIT_PATH` in any demo or production environment.
