# Copyright (c) 2025 Vimal Nakrani
# Licensed under the MIT license.

"""Environment-aware configuration for stillwarm.

Reads environment variables (optionally from a `.env` file) with defaults
and validation. All env vars are prefixed with `STILLWARM_`.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration with env-var override support.

    Loads from environment variables with the STILLWARM_ prefix.
    Optionally reads from a .env file in the working directory.
    """

    model_config = SettingsConfigDict(
        env_prefix="STILLWARM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Silently-ineffective-restore guard thresholds (measured defaults).
    # If the first request after a restore re-prefills more than this
    # fraction of the prompt — or more than this many tokens, whichever is
    # larger — the restore did not take.
    guard_ratio: float = 0.5
    guard_min_tokens: int = 64

    # Server defaults.
    base_url: str = "http://127.0.0.1:8080"
    timeout: float = 600.0

    # Cache management.
    disk_budget_gb: float | None = None
    cache_dir: str = ""


def get_settings() -> Settings:
    """Return the current settings instance."""
    return Settings()
