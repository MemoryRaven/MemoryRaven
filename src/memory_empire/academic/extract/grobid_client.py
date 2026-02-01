from __future__ import annotations

from dataclasses import dataclass

from memory_empire.academic.http import HttpClientFactory, transient_retry
from memory_empire.academic.settings import settings


@dataclass(frozen=True)
class GrobidResult:
    tei_xml: str


class GrobidClient:
    """Client for a self-hosted GROBID server.

    Recommended endpoints:
    - `/api/processFulltextDocument` (TEI with structure)
    - `/api/processHeaderDocument` (header-only)

    Run grobid as docker for dev:
    `docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1`
    """

    def __init__(self, base_url: str | None = None):
        self._client = HttpClientFactory.client(base_url=(base_url or settings.grobid_url))

    async def aclose(self):
        await self._client.aclose()

    @transient_retry()
    async def process_fulltext(
        self, pdf_bytes: bytes, consolidate_citations: bool = True
    ) -> GrobidResult:
        files = {"input": ("paper.pdf", pdf_bytes, "application/pdf")}
        data = {
            "consolidateCitations": "1" if consolidate_citations else "0",
            "includeRawAffiliations": "1",
            "includeRawCitations": "1",
        }
        r = await self._client.post("/api/processFulltextDocument", data=data, files=files)
        r.raise_for_status()
        return GrobidResult(tei_xml=r.text)
