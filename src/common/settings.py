from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORCH_", env_file=".env", extra="ignore")

    env: str = "dev"  # dev | prod
    api_keys: dict[str, str] = Field(
        default_factory=lambda: {"victim": "vk_dev", "counsellor": "ck_dev", "ops": "ok_dev"}
    )
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "EMPTY"
    llm_model_map: dict[str, str] = Field(default_factory=lambda: {"dialogue": "sahayak", "summary": "casewriter"})
    llm_timeout_s: int = 120

    services: dict[str, str] = Field(
        default_factory=lambda: {
            "scoring": "http://localhost:8100",
            "fusion": "http://localhost:8200",
            "forecast": "http://localhost:8300",
            "backend": "http://localhost:8400",
        }
    )
    service_timeout_s: float = 3.0

    policies: dict[str, float] = Field(
        default_factory=lambda: {
            "escalate_score": 0.75,
            "escalate_confidence_floor": 0.45,
            "forecast_escalation_prob": 0.60,
        }
    )
    policy_version: str = "1.0.0"

    checkpoint_dsn: str | None = None
    audit_path: str = "./data/audit/ledger.jsonl"
    rate_limit_rpm: int = 60
    max_concurrent_graphs: int = 6
    max_message_chars: int = 4000
    request_budget_ms: int = 12000
    audit_hmac_key: str = "dev-only-change-me"
    redis_url: str | None = None
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    @field_validator("policies")
    @classmethod
    def _sane_policies(cls, v: dict) -> dict:
        if not (0 < v["escalate_score"] <= 1 and 0 < v["forecast_escalation_prob"] <= 1):
            raise ValueError("policies out of range: escalate_score/forecast_escalation_prob must be in (0, 1]")
        return v

    @model_validator(mode="after")
    def _production_ready(self):
        required_services = {"scoring", "fusion", "forecast", "backend"}
        missing_services = required_services - set(self.services)
        if missing_services:
            missing = ", ".join(sorted(missing_services))
            raise ValueError(f"missing service endpoints: {missing}")

        required_models = {"dialogue", "summary"}
        missing_models = required_models - set(self.llm_model_map)
        if missing_models:
            missing = ", ".join(sorted(missing_models))
            raise ValueError(f"missing LLM model mappings: {missing}")

        if self.env == "prod":
            dev_keys = {"vk_dev", "ck_dev", "ok_dev"}
            configured_keys = set(self.api_keys.values())
            if configured_keys & dev_keys:
                raise ValueError("prod environment cannot use default development API keys")
            if "*" in self.allowed_origins:
                raise ValueError("prod environment cannot allow wildcard CORS origins")
            if not self.checkpoint_dsn:
                raise ValueError("prod environment requires ORCH_CHECKPOINT_DSN")
        return self


settings = Settings()
