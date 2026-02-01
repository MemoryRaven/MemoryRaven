from __future__ import annotations

import argparse
import os
from collections.abc import Iterable

from .embeddings import build_default_embedders
from .indexer import Indexer, StreamWatcher, ensure_dirs
from .router import Router
from .sources import FileGlobSource, WebSource
from .storage_sqlite import SQLiteIndexDB
from .vector_store import build_vector_store
from .web_cache import WebCache


def _parse_sources(specs: list[str], *, web_cache: WebCache) -> Iterable:
    """Parse --source entries.

    Format: kind:value
      - web:https://...
      - conversations:/path/dir (ingests **/*.md, **/*.txt)
      - logs:/path/*.log
    """
    for s in specs:
        if ":" not in s:
            raise SystemExit(f"Invalid --source {s}; expected kind:value")
        kind, val = s.split(":", 1)
        kind = kind.strip().lower()
        val = val.strip()

        if kind == "web":
            yield WebSource(url=val, cache=web_cache)
        elif kind in {"conversations", "memory"}:
            glob_pat = val
            if os.path.isdir(val):
                glob_pat = os.path.join(val, "**/*")
            yield FileGlobSource(path_glob=glob_pat, source_name=kind, content_type="conversation")
        elif kind in {"tools", "tool_outputs"}:
            glob_pat = val
            if os.path.isdir(val):
                glob_pat = os.path.join(val, "**/*")
            yield FileGlobSource(path_glob=glob_pat, source_name=kind, content_type="tool_output")
        elif kind in {"logs", "log"}:
            glob_pat = val
            if os.path.isdir(val):
                glob_pat = os.path.join(val, "**/*.log")
            yield FileGlobSource(path_glob=glob_pat, source_name=kind, content_type="log")
        else:
            # generic file glob
            yield FileGlobSource(path_glob=val, source_name=kind, content_type="generic")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vector_indexing")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_batch = sub.add_parser("batch")
    ap_batch.add_argument("--db", required=True)
    ap_batch.add_argument("--vector-dir", default=None)
    ap_batch.add_argument("--namespace", default="default")
    ap_batch.add_argument("--source", action="append", default=[])

    ap_stream = sub.add_parser("stream")
    ap_stream.add_argument("--db", required=True)
    ap_stream.add_argument("--vector-dir", default=None)
    ap_stream.add_argument("--namespace", default="default")
    ap_stream.add_argument("--watch", required=True)

    args = ap.parse_args(argv)

    ensure_dirs(os.path.dirname(args.db), args.vector_dir)

    web_cache = WebCache(
        db_path=os.path.join(os.path.dirname(args.db), "web_cache.db"),
        body_dir=os.path.join(os.path.dirname(args.db), "web_cache"),
    )

    db = SQLiteIndexDB(path=args.db)
    vector_store = build_vector_store(args.vector_dir)
    embedders = build_default_embedders()
    router = Router(default_namespace=args.namespace)
    indexer = Indexer(db=db, vector_store=vector_store, embedders=embedders, router=router)

    if args.cmd == "batch":
        docs = []
        for src in _parse_sources(args.source, web_cache=web_cache):
            docs.extend(list(src.iter_documents()))
        stats = indexer.index_documents(docs, namespace=args.namespace)
        print(stats)
        return 0

    if args.cmd == "stream":
        watcher = StreamWatcher(path=args.watch)
        for doc in watcher.iter_documents():
            indexer.index_documents([doc], namespace=args.namespace)
        return 0

    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
