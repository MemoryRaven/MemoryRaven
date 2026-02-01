from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalization used for hashing/dedup.

    Keeps meaning but removes irrelevant variance.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def strip_boilerplate_lines(text: str) -> str:
    """Heuristic cleanup for logs/tool output."""
    lines = text.splitlines()
    cleaned: list[str] = []
    for ln in lines:
        if not ln.strip():
            continue
        # remove very long repeating separator lines
        if len(ln) > 20 and len(set(ln.strip())) <= 3:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)


def guess_language(text: str) -> str:
    """Tiny heuristic; replace with fastText/langid later."""
    # If mostly ASCII, call it 'en' as a pragmatic default.
    if not text:
        return "unknown"
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(1, len(text))
    return "en" if ascii_ratio > 0.9 else "unknown"
