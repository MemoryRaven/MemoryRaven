from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MemoryServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMORY_SERVICE_", extra="ignore")

    # HTTP
    bind_host: str = "0.0.0.0"
    bind_port: int = 8088

    # Logging
    log_level: str = "INFO"

    # Auth
    api_key: str | None = Field(default=None, description="If set, require X-API-Key")

    # Postgres
    postgres_dsn: str = Field(
        default="postgresql://memory:memory@localhost:5432/memory",
        description="asyncpg DSN",
    )

    # Redis queue
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "memory:embed:queue"

    # Vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "memory_events"

    # Graph DB (Neo4j)
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"

    # Embeddings
    embedding_dim: int = 384
    st_model: str | None = Field(
        default=None,
        description="sentence-transformers model name (optional). If unset, use stub embedder.",
    )


settings = MemoryServiceSettings()
