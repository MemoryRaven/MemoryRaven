from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from memory_empire.academic.models import CitationEdge


def extract_bibliography_dois_from_grobid_tei(tei_xml: str) -> list[str]:
    """Extract candidate DOIs from GROBID TEI.

    This is a pragmatic approach: look for <idno type="DOI"> and DOI-like strings.
    For best results, also store the full biblStruct and attempt title/author matching.
    """

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    root = ET.fromstring(tei_xml)
    dois: list[str] = []

    for idno in root.findall(".//tei:listBibl//tei:biblStruct//tei:idno", ns):
        if (idno.attrib.get("type") or "").lower() == "doi" and idno.text:
            dois.append(_normalize_doi(idno.text))

    # Fallback regex search
    if not dois:
        text = "".join(root.itertext())
        for m in re.finditer(r"10\.[0-9]{4,9}/[^\s\"<>]+", text):
            dois.append(_normalize_doi(m.group(0)))

    return sorted({d for d in dois if d})


def _normalize_doi(x: str) -> str:
    x = x.strip()
    x = re.sub(r"^https?://(dx\.)?doi\.org/", "", x, flags=re.I)
    return x


def build_citation_edges(citing_key: str, cited_dois: list[str]) -> list[CitationEdge]:
    return [CitationEdge(citing_key=citing_key, cited_key=f"doi:{doi}") for doi in cited_dois]
