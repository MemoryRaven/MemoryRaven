from __future__ import annotations

import math
import re

from .util_text import normalize_text

_URL_RE = re.compile(r"https?://\S+")


def quality_score(text: str, *, content_type: str = "generic") -> float:
    """0..1 quality heuristic used for filtering and routing.

    Replace/augment with a learned model later.
    """
    t = normalize_text(text)
    if not t:
        return 0.0

    length = len(t)
    # penalize extremely short chunks
    # NOTE: tuned so that small but meaningful notes (~30-80 chars) can still pass
    # the default indexer min_quality threshold.
    length_score = 1 - math.exp(-length / 300)

    # penalize if mostly URL soup
    url_count = len(_URL_RE.findall(t))
    url_penalty = min(0.7, url_count / 15)

    # penalize if too repetitive
    tokens = t.lower().split(" ")
    uniq = len(set(tokens))
    rep_ratio = 1 - (uniq / max(1, len(tokens)))
    rep_penalty = min(0.5, rep_ratio)

    base = length_score * (1 - url_penalty) * (1 - rep_penalty)

    # content-type adjustments
    if content_type in {"log", "tool_output"}:
        base *= 0.9  # logs are noisy
    if content_type in {"paper", "pdf"}:
        base *= 1.05

    return max(0.0, min(1.0, base))


def should_index(text: str, *, min_quality: float = 0.25) -> bool:
    return quality_score(text) >= min_quality
