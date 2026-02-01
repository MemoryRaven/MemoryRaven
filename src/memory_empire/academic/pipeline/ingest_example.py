from __future__ import annotations

import asyncio

from memory_empire.academic.clients.semantic_scholar import SemanticScholarClient
from memory_empire.academic.pipeline.keys import canonical_paper_key


async def main():
    s2 = SemanticScholarClient()
    try:
        papers = await s2.search("retrieval augmented generation", limit=5)
        for p in papers:
            key = canonical_paper_key(p.ids)
            print(key, p.title)
    finally:
        await s2.aclose()


if __name__ == "__main__":
    asyncio.run(main())
