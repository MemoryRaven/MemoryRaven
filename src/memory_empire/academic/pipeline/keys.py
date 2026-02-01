from __future__ import annotations

from memory_empire.academic.models import Identifier


def canonical_paper_key(ids: Identifier) -> str:
    """Create a stable primary key for internal use.

    Priority:
    1) DOI (normalized, lowercase)
    2) arXiv
    3) Semantic Scholar PaperId

    For scale: store multiple alias keys -> canonical key table.
    """

    if ids.doi:
        return f"doi:{ids.doi.lower()}"
    if ids.arxiv_id:
        return f"arxiv:{ids.arxiv_id}"
    if ids.s2_paper_id:
        return f"s2:{ids.s2_paper_id}"
    raise ValueError("No usable identifier to build paper_key")
