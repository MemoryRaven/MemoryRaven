from __future__ import annotations

from memory_empire.academic.models import Summary


class Summarizer:
    """Interface for summarization + key findings extraction.

    Implementations may use:
    - LLM API (preferred for quality)
    - local models
    - extractive summarization (fallback)

    Keep it deterministic & cache outputs by (paper_key, model_version).
    """

    async def summarize(
        self, paper_key: str, title: str, abstract: str | None, fulltext: str | None
    ) -> Summary:
        # Placeholder: return a minimal structure.
        # Main agent should wire this to the project's LLM stack.
        tldr = None
        if abstract:
            tldr = abstract.split(".")[0].strip()[:300] or None
        return Summary(paper_key=paper_key, tldr=tldr)
