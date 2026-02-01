from __future__ import annotations

from fastapi import FastAPI

from memory_empire.academic.clients.arxiv import ArxivClient
from memory_empire.academic.clients.semantic_scholar import SemanticScholarClient

app = FastAPI(title="AKS - Academic Knowledge System", version="0.1.0")


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/search/arxiv")
async def search_arxiv(q: str, max_results: int = 10):
    client = ArxivClient()
    try:
        papers = await client.search(query=q, max_results=max_results)
        return {"count": len(papers), "results": [p.model_dump() for p in papers]}
    finally:
        await client.aclose()


@app.get("/search/semantic_scholar")
async def search_semantic_scholar(q: str, limit: int = 10, offset: int = 0):
    client = SemanticScholarClient()
    try:
        papers = await client.search(query=q, limit=limit, offset=offset)
        return {"count": len(papers), "results": [p.model_dump() for p in papers]}
    finally:
        await client.aclose()
