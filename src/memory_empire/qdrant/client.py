from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QdrantConfig:
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    prefer_grpc: bool = False
    timeout_s: float = 10.0


def build_qdrant_client(cfg: QdrantConfig):
    """Create a qdrant_client.QdrantClient with lazy import.

    Keeps memory-empire usable without qdrant-client installed.
    """

    if cfg.timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")

    try:
        from qdrant_client import QdrantClient
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "qdrant-client is not installed. Install with: pip install qdrant-client"
        ) from e

    return QdrantClient(
        url=cfg.url,
        api_key=cfg.api_key,
        prefer_grpc=cfg.prefer_grpc,
        timeout=cfg.timeout_s,
    )
