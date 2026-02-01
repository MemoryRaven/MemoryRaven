from __future__ import annotations

import glob
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass

from .types import Document, SourceRef
from .util_text import strip_boilerplate_lines
from .web_cache import WebCache


class Source:
    def iter_documents(self) -> Iterable[Document]:
        raise NotImplementedError


@dataclass
class WebSource(Source):
    url: str
    cache: WebCache

    def iter_documents(self) -> Iterable[Document]:
        info = self.cache.fetch(self.url)
        text = self.cache.load_text(self.url)
        ref = SourceRef(source="web", uri=self.url, revision=info.get("content_sha256"))
        yield Document(
            ref=ref,
            content_type="web",
            text=text,
            media_type="text/html",
            fetched_at=time.time(),
            title=None,
            extra={"web": info},
        )


@dataclass
class FileGlobSource(Source):
    """Ingest files from a directory/glob.

    Used for conversations, tool outputs, markdown notes, etc.
    """

    path_glob: str
    source_name: str
    content_type: str

    def iter_documents(self) -> Iterable[Document]:
        paths = sorted(glob.glob(self.path_glob, recursive=True))
        for p in paths:
            if os.path.isdir(p):
                continue
            try:
                st = os.stat(p)
            except FileNotFoundError:
                continue
            # Use mtime+size as a basic revision.
            rev = f"mtime:{st.st_mtime_ns}-size:{st.st_size}"
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()

            if self.content_type in {"log", "tool_output"}:
                text = strip_boilerplate_lines(text)

            yield Document(
                ref=SourceRef(source=self.source_name, uri=p, revision=rev),
                content_type=self.content_type,  # type: ignore
                text=text,
                media_type="text/plain",
                created_at=st.st_mtime,
                fetched_at=time.time(),
                title=os.path.basename(p),
                extra={"path": p},
            )


@dataclass
class JSONLToolOutputSource(Source):
    """Ingest JSONL where each line is a tool result.

    Expected fields (best-effort):
      - timestamp
      - tool
      - input
      - output
    """

    path: str

    def iter_documents(self) -> Iterable[Document]:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return
        rev = f"mtime:{st.st_mtime_ns}-size:{st.st_size}"

        with open(self.path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get("output") or obj.get("text") or json.dumps(obj, ensure_ascii=False)
                ts = obj.get("timestamp")
                if isinstance(ts, (int, float)):
                    created_at: float | None = float(ts)
                else:
                    created_at = None
                uri = f"{self.path}::line:{i}"
                yield Document(
                    ref=SourceRef(source="tool_outputs", uri=uri, revision=rev),
                    content_type="tool_output",
                    text=strip_boilerplate_lines(str(text)),
                    media_type="application/json",
                    created_at=created_at,
                    fetched_at=time.time(),
                    title=obj.get("tool"),
                    extra={"raw": obj},
                )
