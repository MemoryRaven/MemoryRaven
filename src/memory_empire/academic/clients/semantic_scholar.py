from __future__ import annotations

from datetime import date

from memory_empire.academic.http import HttpClientFactory, transient_retry
from memory_empire.academic.models import Author, Identifier, PaperRecord, Venue
from memory_empire.academic.settings import settings


class SemanticScholarClient:
    """Semantic Scholar Graph API client.

    Docs: https://api.semanticscholar.org/

    Notes for scale:
    - Prefer bulk datasets (S2 Open Research Corpus) where possible.
    - Respect rate limits; cache aggressively.
    """

    def __init__(self):
        headers = {}
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key
        self._client = HttpClientFactory.client(
            base_url="https://api.semanticscholar.org/graph/v1", headers=headers
        )

    async def aclose(self):
        await self._client.aclose()

    @transient_retry()
    async def paper(
        self,
        paper_id: str,
        fields: str = "title,abstract,year,authors,venue,publicationDate,externalIds,url,openAccessPdf,citationCount,referenceCount",
    ) -> PaperRecord:
        r = await self._client.get(f"/paper/{paper_id}", params={"fields": fields})
        r.raise_for_status()
        d = r.json()
        return self._to_record(d)

    @transient_retry()
    async def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        fields: str = "title,abstract,year,authors,venue,publicationDate,externalIds,url,openAccessPdf",
    ) -> list[PaperRecord]:
        r = await self._client.get(
            "/paper/search",
            params={"query": query, "limit": limit, "offset": offset, "fields": fields},
        )
        r.raise_for_status()
        items = r.json().get("data", [])
        return [self._to_record(x) for x in items]

    def _to_record(self, d: dict) -> PaperRecord:
        ext = d.get("externalIds") or {}
        pub_date: date | None = None
        if d.get("publicationDate"):
            try:
                pub_date = date.fromisoformat(d["publicationDate"])
            except Exception:
                pass

        authors = []
        for a in d.get("authors") or []:
            if a.get("name"):
                authors.append(Author(name=a["name"], author_id=a.get("authorId")))

        pdf_url = None
        oapdf = d.get("openAccessPdf")
        if isinstance(oapdf, dict):
            pdf_url = oapdf.get("url")

        venue_name = d.get("venue")
        venue = Venue(name=venue_name, type="other") if venue_name else None

        return PaperRecord(
            ids=Identifier(
                doi=ext.get("DOI"),
                arxiv_id=ext.get("ArXiv"),
                s2_paper_id=d.get("paperId"),
                pmid=ext.get("PubMed"),
                pmcid=ext.get("PubMedCentral"),
            ),
            title=d.get("title") or "",
            abstract=d.get("abstract"),
            authors=authors,
            venue=venue,
            published_date=pub_date,
            year=d.get("year"),
            landing_page_url=d.get("url"),
            pdf_url=pdf_url,
            source_payloads={"semantic_scholar": d},
        )
