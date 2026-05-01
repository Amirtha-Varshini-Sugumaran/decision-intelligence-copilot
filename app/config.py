"""Application configuration."""

from functools import lru_cache
import os
from pathlib import Path


def _load_env_file(path: Path = Path(".env")) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from a local .env file."""

    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class Settings:
    """Runtime settings loaded from environment variables or .env."""

    def __init__(
        self,
        app_name: str | None = None,
        openai_api_key: str | None = None,
        openai_model: str | None = None,
        data_dir: Path | str | None = None,
        reports_dir: Path | str | None = None,
        max_upload_mb: int | None = None,
    ) -> None:
        env_file = _load_env_file()
        self.app_name = app_name or os.getenv("APP_NAME") or env_file.get("APP_NAME", "Decision Intelligence Copilot")
        self.openai_api_key = (
            openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY") or env_file.get("OPENAI_API_KEY")
        )
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL") or env_file.get("OPENAI_MODEL", "gpt-4.1-mini")
        self.data_dir = Path(data_dir or os.getenv("DATA_DIR") or env_file.get("DATA_DIR", "data"))
        self.reports_dir = Path(reports_dir or os.getenv("REPORTS_DIR") or env_file.get("REPORTS_DIR", "reports"))
        self.max_upload_mb = int(max_upload_mb or os.getenv("MAX_UPLOAD_MB") or env_file.get("MAX_UPLOAD_MB", "10"))


@lru_cache
def get_settings() -> Settings:
    """Return cached settings for dependency injection."""

    return Settings()
