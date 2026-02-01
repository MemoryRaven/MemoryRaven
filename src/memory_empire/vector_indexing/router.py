from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .types import Chunk


@dataclass
class RoutingDecision:
    namespace: str
    collection_hint: str
    extra: dict[str, Any]


@dataclass
class Router:
    """Namespace organization and smart routing logic.

    Philosophy:
    - namespace is a high-level partition (project, domain, channel)
    - collection_hint selects vector store collection (often content_type + model)
    """

    default_namespace: str = "default"

    def route(self, chunk: Chunk, *, namespace_override: str | None = None) -> RoutingDecision:
        ns = namespace_override or self.default_namespace

        # Web routing by domain
        domain = (chunk.metadata or {}).get("domain")
        if domain:
            ns = re.sub(r"[^a-zA-Z0-9_-]+", "_", domain)[:60]

        ct = chunk.content_type
        collection = f"{ct}"
        return RoutingDecision(namespace=ns, collection_hint=collection, extra={})
