from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .util_text import normalize_text


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _token_hash(token: str) -> int:
    h = hashlib.md5(token.encode("utf-8")).digest()  # stable across runs
    return int.from_bytes(h[:8], "big", signed=False)


def simhash(text: str, *, bits: int = 64) -> str:
    """Near-duplicate fingerprint (simple simhash).

    Not cryptographic. Useful for detecting high-overlap chunks.
    """
    t = normalize_text(text)
    if not t:
        return "0" * (bits // 4)

    # Basic tokenization.
    tokens = [tok for tok in t.lower().split(" ") if tok]
    v = [0] * bits
    for tok in tokens:
        x = _token_hash(tok)
        for i in range(bits):
            bit = (x >> i) & 1
            v[i] += 1 if bit else -1

    out = 0
    for i, w in enumerate(v):
        if w > 0:
            out |= 1 << i
    return f"{out:0{bits // 4}x}"


def hamming_distance_hex(a_hex: str, b_hex: str) -> int:
    a = int(a_hex, 16)
    b = int(b_hex, 16)
    return (a ^ b).bit_count()


@dataclass(frozen=True)
class Fingerprints:
    sha256: str
    simhash64: str


def fingerprint(text: str) -> Fingerprints:
    return Fingerprints(sha256=sha256_text(text), simhash64=simhash(text, bits=64))
