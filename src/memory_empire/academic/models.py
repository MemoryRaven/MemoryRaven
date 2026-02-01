from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class Identifier(BaseModel):
    """Canonical identifiers. Prefer DOI when available."""

    doi: str | None = None
    arxiv_id: str | None = None
    s2_paper_id: str | None = None
    pmid: str | None = None
    pmcid: str | None = None


class Author(BaseModel):
    name: str
    author_id: str | None = None
    affiliations: list[str] = Field(default_factory=list)


class Venue(BaseModel):
    name: str | None = None
    type: Literal["journal", "conference", "preprint", "book", "other"] = "other"


class PaperRecord(BaseModel):
    ids: Identifier = Field(default_factory=Identifier)
    title: str
    abstract: str | None = None
    authors: list[Author] = Field(default_factory=list)
    venue: Venue | None = None
    published_date: date | None = None
    year: int | None = None

    # URLs
    landing_page_url: str | None = None
    pdf_url: str | None = None

    # Raw provider payloads for audit/debug (optional)
    source_payloads: dict[str, Any] = Field(default_factory=dict)


class FullTextChunk(BaseModel):
    paper_key: str
    chunk_id: str
    text: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None


class CitationEdge(BaseModel):
    citing_key: str
    cited_key: str
    context: str | None = None
    intent: str | None = None


class Summary(BaseModel):
    paper_key: str
    tldr: str | None = None
    abstract_summary: str | None = None
    key_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)


class Classification(BaseModel):
    paper_key: str
    domains: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
