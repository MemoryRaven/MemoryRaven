from __future__ import annotations

# Standard library imports
import os
from dataclasses import dataclass
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:  # noqa: D401
        """Return one vector per text."""


@dataclass
class MultiEmbedConfig:
    # two families: semantic (for retrieval) + contextual (for re-ranking / context matching)
    semantic_model: str = "local-stub-semantic"
    contextual_model: str = "local-stub-contextual"


class StubEmbedder:
    """Deterministic no-dependency embedder.

    This is NOT useful for real retrieval, but keeps the pipeline runnable.
    Replace with SentenceTransformersEmbedder or API-based embedders.
    """

    def __init__(self, name: str, dim: int = 64):
        self.name = name
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            # Simple hashing into a fixed vector.
            v = [0.0] * self.dim
            for i, ch in enumerate(t.encode("utf-8")):
                v[(i + ch) % self.dim] += 1.0
            # L2 normalize.
            norm = sum(x * x for x in v) ** 0.5
            if norm:
                v = [x / norm for x in v]
            out.append(v)
        return out


class SentenceTransformersEmbedder:
    def __init__(self, model_name: str):
        self.name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "sentence-transformers not installed; pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def build_default_embedders(cfg: MultiEmbedConfig | None = None) -> dict[str, Embedder]:
    cfg = cfg or MultiEmbedConfig()

    # Prefer sentence-transformers if explicitly requested.
    embedders: dict[str, Embedder] = {}

    if os.getenv("VECTOR_INDEX_SEMANTIC_ST_MODEL"):
        embedders["semantic"] = SentenceTransformersEmbedder(
            os.environ["VECTOR_INDEX_SEMANTIC_ST_MODEL"]
        )
    else:
        embedders["semantic"] = StubEmbedder(cfg.semantic_model, dim=128)

    if os.getenv("VECTOR_INDEX_CONTEXTUAL_ST_MODEL"):
        embedders["contextual"] = SentenceTransformersEmbedder(
            os.environ["VECTOR_INDEX_CONTEXTUAL_ST_MODEL"]
        )
    else:
        embedders["contextual"] = StubEmbedder(cfg.contextual_model, dim=96)

    return embedders
