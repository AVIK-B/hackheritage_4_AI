setup:    ; pip install -r requirements-orch.txt -r requirements-dev.txt
lint:     ; ruff check src tests
test:     ; pytest -q
corpus:   ; python src/data/build_case_summary_corpus.py --n 8000 && python src/data/build_dialogue_corpus.py
up:       ; docker compose -f docker-compose.cpu.yml up -d --build
down:     ; docker compose -f docker-compose.cpu.yml down
smoke:    ; bash scripts/smoke.sh
trace:    ; python scripts/demo_trace.py
verify:   ; python scripts/verify_ledger.py