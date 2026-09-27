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

    log_level: str = os.environ.get("LOG_LEVEL", "INFO")


settings = Settings()
