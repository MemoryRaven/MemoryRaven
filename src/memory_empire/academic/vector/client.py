from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VectorPoint:
    id: str
    vector: list[float]
    payload: dict


class VectorDbClient:
    """Abstract interface expected by AKS.

    Main agent should implement this against the project's existing vector DB.
    """

    async def upsert(self, points: list[VectorPoint]) -> None:  # pragma: no cover
        raise NotImplementedError

    async def query(
        self, vector: list[float], k: int = 10, filter: dict | None = None
    ) -> list[dict]:  # pragma: no cover
        raise NotImplementedError
