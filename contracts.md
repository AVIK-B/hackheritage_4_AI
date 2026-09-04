# Service Contracts

This document records the HTTP contracts assumed by the orchestrator. All JSON
examples are minimal but sufficient for the current graph.

## Authentication

Protected orchestrator endpoints require:

```http
X-API-Key: <role key>
```

Roles are configured through `ORCH_API_KEYS`.

## Orchestrator API

### POST `/v1/interactions`

Role: `victim`

Request:

```json
{
  "case_id": "CASE-1001",
  "channel": "pwa",
  "message": "aaj thoda better feel kar raha hoon",
  "language": "hi"
}
```

Response:

```json
{
  "thread_id": "hex-thread-id",
  "reply": "victim-facing message",
  "status": "completed",
  "audit_ref": "sha256-record-hash"
}
```

`status` is `awaiting_counsellor` when the graph is interrupted before high-risk
dispatch and a counsellor must resume it.

### POST `/v1/interactions/stream`

Role: `victim`

Same request body as `/v1/interactions`. Response is Server-Sent Events:

```text
data: {"delta":"partial text"}

data: {"final":{"thread_id":"...","status":"completed","audit_ref":"...","reply":"..."}}
```

### POST `/v1/counsellor/{thread_id}/resume`

Role: `counsellor`

Request:

```json
{
  "approved": true,
  "notes": "counsellor reviewed and approved escalation"
}
```

Response:

```json
{
  "status": "completed",
  "alert_id": "ALERT-123",
  "audit_ref": "sha256-record-hash"
}
```

### GET `/v1/threads/{thread_id}/trace`

Role: `counsellor`

Returns decision, fusion, forecast, signal names, graph errors, and next graph node.

### GET `/v1/audit/verify`

Role: `ops`

Returns ledger validity, record count, and last hash.

### GET `/healthz`

Public liveness endpoint.

### GET `/readyz`

Role: `ops`

Returns LLM health and graph readiness.

### GET `/metrics`

Role: `ops`

Prometheus metrics.

## Backend Service

Base URL: `ORCH_SERVICES.backend`

### GET `/v1/case/{case_id}/context`

Response:

```json
{
  "case_id": "CASE-1001",
  "language": "hi",
  "case_stage": "hearing_scheduled",
  "days_to_next_hearing": 5,
  "recent_messages": [
    {"role": "user", "content": "previous check-in"},
    {"role": "assistant", "content": "previous reply"}
  ],
  "recent_checkins": [{"score": 0.4, "ts": "2026-09-04T10:00:00Z"}],
  "score_history": [0.2, 0.3, 0.5]
}
```

The orchestrator degrades safely if this endpoint is unavailable.

### POST `/v1/alerts`

Request:

```json
{
  "case_id": "CASE-1001",
  "severity": "high",
  "reasons": ["composite 0.82 >= 0.75"],
  "summary_text": "grounded counsellor summary"
}
```

Response:

```json
{
  "alert_id": "ALERT-123"
}
```

If dispatch fails, the orchestrator writes a dead-letter entry to the audit ledger.

## Scoring Service

Base URL: `ORCH_SERVICES.scoring`

Each scoring endpoint accepts:

```json
{
  "text": "scrubbed user message",
  "audio_ref": null,
  "lang": "en"
}
```

Expected response shape:

```json
{
  "score": 0.73,
  "prob": 0.73,
  "label": "elevated",
  "confidence": 0.81
}
```

Endpoints:

- `POST /v1/signals/text`
- `POST /v1/signals/threat`
- `POST /v1/signals/voice`

## Fusion Service

Base URL: `ORCH_SERVICES.fusion`

### POST `/v1/fusion`

Request:

```json
{
  "sentiment": {"score": 0.4},
  "threat": {"score": 0.8},
  "voice_stress": null,
  "engagement": [],
  "case_stage": "hearing_scheduled"
}
```

Response:

```json
{
  "composite_score": 0.78,
  "confidence": 0.62,
  "top_signals": ["threat", "sentiment"]
}
```

## Forecast Service

Base URL: `ORCH_SERVICES.forecast`

### POST `/v1/forecast`

Request:

```json
{
  "case_id": "CASE-1001",
  "score_history": [0.2, 0.3, 0.5]
}
```

Response:

```json
{
  "p_escalation": 0.64,
  "horizon_days": 7
}
```

## LLM Endpoint

Base URL: `ORCH_LLM_BASE_URL`

The endpoint must be OpenAI-compatible and support:

- `GET /models`
- `POST /chat/completions`
- `POST /chat/completions` with `stream=true`

Configured model names:

- `sahayak` for victim-facing dialogue
- `casewriter` for grounded case summaries
