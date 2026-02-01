from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    semantic_scholar_api_key: str | None = None
    core_api_key: str | None = None
    crossref_mailto: str | None = None

    grobid_url: str = "http://localhost:8070"

    vectordb_url: str | None = None
    vectordb_api_key: str | None = None
    vectordb_namespace: str = "aks"


settings = Settings()
