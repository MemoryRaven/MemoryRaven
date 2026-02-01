from __future__ import annotations

from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pydantic-settings is required. Install with: pip install pydantic-settings"
    ) from e


class MemoryEmpireSettings(BaseSettings):
    """Unified configuration for Memory Empire.

    Environment variables are prefixed with MEMORY_EMPIRE_.
    """

    model_config = SettingsConfigDict(env_prefix="MEMORY_EMPIRE_", extra="ignore")

    # --- Core ---
    log_level: str = Field(default="INFO", description="Python logging level")

    # --- Memory-OS ---
    memory_db_path: str = Field(default="~/.claude_memory/memory.db")

    # --- Vector Indexing ---
    vector_store: str = Field(default="jsonl", description="jsonl|chroma|qdrant")
    vector_dir: str | None = Field(default=None, description="For chroma/jsonl persistence")

    # --- Qdrant ---
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)


settings = MemoryEmpireSettings()
