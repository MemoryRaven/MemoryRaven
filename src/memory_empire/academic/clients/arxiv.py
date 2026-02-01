from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date
from urllib.parse import quote

from memory_empire.academic.http import HttpClientFactory, transient_retry
from memory_empire.academic.models import Author, Identifier, PaperRecord, Venue

_ARXIV_API = "https://export.arxiv.org/api/query"


def _strip(s: str | None) -> str | None:
    return s.strip() if s is not None else None


def _arxiv_id_from_id_url(id_url: str) -> str | None:
    # examples: http://arxiv.org/abs/2101.00001v2
    m = re.search(r"/abs/([^/]+)$", id_url)
    return m.group(1) if m else None


class ArxivClient:
    """Minimal arXiv API client.

    arXiv's API is Atom XML. For scale, use bulk data / S3 mirror.
    """

    def __init__(self):
        self._client = HttpClientFactory.client(base_url=_ARXIV_API)

    async def aclose(self):
        await self._client.aclose()

    @transient_retry()
    async def search(self, query: str, start: int = 0, max_results: int = 100) -> list[PaperRecord]:
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        r = await self._client.get("", params=params)
        r.raise_for_status()
        return self._parse_feed(r.text)

    def _parse_feed(self, xml_text: str) -> list[PaperRecord]:
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(xml_text)
        papers: list[PaperRecord] = []
        for entry in root.findall("atom:entry", ns):
            title = _strip(entry.findtext("atom:title", default="", namespaces=ns)) or ""
            abstract = _strip(entry.findtext("atom:summary", default=None, namespaces=ns))
            id_url = entry.findtext("atom:id", default="", namespaces=ns)
            arxiv_id = _arxiv_id_from_id_url(id_url)

            published = entry.findtext("atom:published", default=None, namespaces=ns)
            published_date: date | None = None
            year: int | None = None
            if published:
                try:
                    published_date = date.fromisoformat(published[:10])
                    year = published_date.year
                except Exception:
                    pass

            authors = []
            for a in entry.findall("atom:author", ns):
                name = _strip(a.findtext("atom:name", default="", namespaces=ns)) or ""
                if name:
                    authors.append(Author(name=name))

            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href")

            papers.append(
                PaperRecord(
                    ids=Identifier(arxiv_id=arxiv_id),
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    venue=Venue(name="arXiv", type="preprint"),
                    published_date=published_date,
                    year=year,
                    landing_page_url=id_url,
                    pdf_url=pdf_url,
                    source_payloads={"arxiv": {"id": id_url}},
                )
            )
        return papers

    @staticmethod
    def arxiv_query_all(words: str) -> str:
        """Helper for building simple arXiv queries."""
        return f"all:{quote(words)}"
