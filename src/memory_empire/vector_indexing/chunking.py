from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .types import Chunk, Document
from .util_text import normalize_text


@dataclass
class ChunkingConfig:
    # target sizes are soft; overlap helps recall
    target_chars: int = 1200
    min_chars: int = 300
    overlap_chars: int = 150


def _sliding_window(text: str, *, target: int, overlap: int) -> list[tuple[int, int, str]]:
    t = text
    n = len(t)
    if n <= target:
        return [(0, n, t)]

    out: list[tuple[int, int, str]] = []
    start = 0
    while start < n:
        end = min(n, start + target)
        # try to end on a newline boundary
        nl = t.rfind("\n", start, end)
        if nl != -1 and (end - nl) < 200:
            end = nl
        seg = t[start:end].strip()
        if seg:
            out.append((start, end, seg))
        if end == n:
            break
        start = max(0, end - overlap)
    return out


def chunk_document(doc: Document, cfg: ChunkingConfig | None = None) -> Iterable[Chunk]:
    cfg = cfg or ChunkingConfig()
    text = doc.text

    # Type-specific adjustments.
    if doc.content_type in {"conversation"}:
        target = 1400
        overlap = 100
    elif doc.content_type in {"paper", "pdf"}:
        target = 1800
        overlap = 200
    elif doc.content_type in {"log", "tool_output"}:
        target = 1000
        overlap = 80
    else:
        target = cfg.target_chars
        overlap = cfg.overlap_chars

    text = normalize_text(text)
    windows = _sliding_window(text, target=target, overlap=overlap)

    for i, (start, end, seg) in enumerate(windows):
        if len(seg) < cfg.min_chars and i != 0 and i != (len(windows) - 1):
            continue
        yield Chunk(
            doc_ref=doc.ref,
            chunk_id=f"{i:04d}",
            content_type=doc.content_type,
            text=seg,
            start=start,
            end=end,
            metadata={
                "title": doc.title,
                "content_type": doc.content_type,
                "uri": doc.ref.uri,
                "source": doc.ref.source,
            },
        )
