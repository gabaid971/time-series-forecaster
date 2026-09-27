"""Settings read from environment variables (all optional)."""

import os
from dataclasses import dataclass, field


def _origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@dataclass
class Settings:
    # The API is public (demo app): protection is about limiting abuse, not access.
    # Frontend origins allowed by browsers (CORS), comma-separated.
    allowed_origins: list[str] = field(default_factory=_origins)

    # Request size limits
    max_body_mb: float = float(os.environ.get("MAX_BODY_MB", "20"))
    max_rows: int = int(os.environ.get("MAX_ROWS", "100000"))
    max_models: int = int(os.environ.get("MAX_MODELS", "10"))

    # Requests per minute and per client IP
    analyze_rate_limit: int = int(os.environ.get("ANALYZE_RATE_LIMIT_PER_MINUTE", "30"))
    train_rate_limit: int = int(os.environ.get("TRAIN_RATE_LIMIT_PER_MINUTE", "10"))

    # Training runs one request at a time (small instance: CPU and memory), within a time budget
    max_concurrent_trainings: int = int(os.environ.get("MAX_CONCURRENT_TRAININGS", "1"))
    train_queue_timeout_s: float = float(os.environ.get("TRAIN_QUEUE_TIMEOUT_S", "30"))
    train_time_budget_s: float = float(os.environ.get("TRAIN_TIME_BUDGET_S", "60"))

    log_level: str = os.environ.get("LOG_LEVEL", "INFO")


settings = Settings()
