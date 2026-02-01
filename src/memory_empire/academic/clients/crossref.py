from __future__ import annotations

from datetime import date

from memory_empire.academic.http import HttpClientFactory, transient_retry
from memory_empire.academic.models import Author, Identifier, PaperRecord, Venue
from memory_empire.academic.settings import settings


class CrossrefClient:
    """Crossref REST API client.

    Docs: https://api.crossref.org/

    Use `mailto` for polite pool.
    """

    def __init__(self):
        headers = {"User-Agent": "aks/0.1 (mailto:%s)" % (settings.crossref_mailto or "")}
        self._client = HttpClientFactory.client(
            base_url="https://api.crossref.org", headers=headers
        )

    async def aclose(self):
        await self._client.aclose()

    @transient_retry()
    async def works(self, query: str, rows: int = 20, offset: int = 0) -> list[PaperRecord]:
        r = await self._client.get(
            "/works", params={"query": query, "rows": rows, "offset": offset}
        )
        r.raise_for_status()
        items = (r.json().get("message") or {}).get("items") or []
        return [self._to_record(x) for x in items]

    def _to_record(self, d: dict) -> PaperRecord:
        doi = d.get("DOI")
        title = (d.get("title") or [""])[0] or ""
        abstract = d.get("abstract")

        authors = []
        for a in d.get("author") or []:
            name = " ".join([x for x in [a.get("given"), a.get("family")] if x])
            if name:
                authors.append(
                    Author(
                        name=name,
                        affiliations=[
                            aff.get("name") for aff in a.get("affiliation") or [] if aff.get("name")
                        ],
                    )
                )

        pub_date = None
        year = None
        issued = (d.get("issued") or {}).get("date-parts")
        if issued and issued[0]:
            parts = issued[0]
            try:
                year = int(parts[0])
                if len(parts) >= 3:
                    pub_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
            except Exception:
                pass

        container = (d.get("container-title") or [None])[0]
        venue = Venue(name=container, type="journal") if container else None

        url = d.get("URL")

        return PaperRecord(
            ids=Identifier(doi=doi),
            title=title,
            abstract=abstract,
            authors=authors,
            venue=venue,
            published_date=pub_date,
            year=year,
            landing_page_url=url,
            pdf_url=None,
            source_payloads={"crossref": d},
        )
