from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recommendation:
    paper_key: str
    score: float
    reason: str | None = None


class Recommender:
    """Interface for recommendations.

    At scale, combine:
    - Graph signals: co-citation, bibliographic coupling, PageRank, field-normalized counts
    - Content signals: vector similarity over title/abstract/fulltext embeddings
    - Freshness + user/domain preferences
    """

    async def recommend_for_paper(self, paper_key: str, k: int = 10) -> list[Recommendation]:
        return []
