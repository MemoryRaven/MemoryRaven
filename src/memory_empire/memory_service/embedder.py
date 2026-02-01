from __future__ import annotations

from dataclasses import dataclass


class Embedder:
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass
class StubEmbedder(Embedder):
    dim: int = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self.dim
            b = t.encode("utf-8", errors="ignore")
            for i, ch in enumerate(b):
                v[(i + ch) % self.dim] += 1.0
            # normalize
            norm = sum(x * x for x in v) ** 0.5
            if norm:
                v = [x / norm for x in v]
            out.append(v)
        return out


class SentenceTransformersEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model_name)
        # infer dim
        try:
            self.dim = int(self._m.get_sentence_embedding_dimension())
        except Exception:
            self.dim = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._m.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def build_embedder(*, st_model: str | None, dim: int) -> Embedder:
    if st_model:
        try:
            emb = SentenceTransformersEmbedder(st_model)
            return emb
        except Exception:
            # fall back
            return StubEmbedder(dim=dim)
    return StubEmbedder(dim=dim)
