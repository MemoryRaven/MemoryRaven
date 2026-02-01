from __future__ import annotations

from datetime import date

from memory_empire.academic.http import HttpClientFactory, transient_retry
from memory_empire.academic.models import Author, Identifier, PaperRecord, Venue
from memory_empire.academic.settings import settings


class CoreClient:
    """CORE API client.

    CORE provides metadata + fulltext links for OA content.
    Docs (v3/v4 vary): https://core.ac.uk/services/api

    This adapter is intentionally minimal and may require endpoint adjustments
    depending on your CORE API version/plan.
    """

    def __init__(self):
        headers = {}
        if settings.core_api_key:
            headers["Authorization"] = f"Bearer {settings.core_api_key}"
        self._client = HttpClientFactory.client(
            base_url="https://api.core.ac.uk/v3", headers=headers
        )

    async def aclose(self):
        await self._client.aclose()

    @transient_retry()
    async def search(self, query: str, limit: int = 20, offset: int = 0) -> list[PaperRecord]:
        r = await self._client.get(
            "/search/works", params={"q": query, "limit": limit, "offset": offset}
        )
        r.raise_for_status()
        items = r.json().get("results") or r.json().get("data") or []
        return [self._to_record(x) for x in items]

    def _to_record(self, d: dict) -> PaperRecord:
        title = d.get("title") or ""
        abstract = d.get("abstract")

        authors = []
        for a in d.get("authors") or []:
            if isinstance(a, str):
                authors.append(Author(name=a))
            elif isinstance(a, dict) and a.get("name"):
                authors.append(Author(name=a["name"]))

        year = d.get("yearPublished") or d.get("year")
        pub_date = None
        if d.get("publishedDate"):
            try:
                pub_date = date.fromisoformat(d["publishedDate"][:10])
            except Exception:
                pass

        doi = d.get("doi")
        pdf_url = None
        for link in d.get("downloadUrl") or d.get("fullTextLink") or []:
            if isinstance(link, str) and link.lower().endswith(".pdf"):
                pdf_url = link

        return PaperRecord(
            ids=Identifier(doi=doi),
            title=title,
            abstract=abstract,
            authors=authors,
            venue=Venue(name=d.get("publisher"), type="other") if d.get("publisher") else None,
            published_date=pub_date,
            year=year,
            landing_page_url=d.get("sourceFulltextUrls", [None])[0]
            if isinstance(d.get("sourceFulltextUrls"), list)
            else d.get("sourceFulltextUrls"),
            pdf_url=pdf_url,
            source_payloads={"core": d},
        )
