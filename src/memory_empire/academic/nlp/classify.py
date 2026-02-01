from __future__ import annotations

from memory_empire.academic.models import Classification


class Classifier:
    """Interface for domain classification + keyword tagging.

    Options:
    - hierarchical label model (e.g., arXiv categories, S2 fields of study)
    - zero-shot via LLM
    - rules for high-precision buckets

    Topic modeling is usually offline (batch) at scale.
    """

    async def classify(
        self, paper_key: str, title: str, abstract: str | None, fulltext: str | None
    ) -> Classification:
        # Placeholder.
        domains = []
        if title.lower().find("transformer") >= 0:
            domains.append("machine learning")
        return Classification(paper_key=paper_key, domains=domains)
