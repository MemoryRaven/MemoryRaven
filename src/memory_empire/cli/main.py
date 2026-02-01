from __future__ import annotations

import argparse
import os
from pathlib import Path

from memory_empire.settings import settings


def _configure_logging() -> None:
    import logging

    level = (settings.log_level or "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def cmd_version() -> int:
    from memory_empire import __version__

    print(__version__)
    return 0


def cmd_memory_ingest_clawdbot(args: argparse.Namespace) -> int:
    _configure_logging()
    from memory_empire.memory_os import MemoryBridge, MemoryOS

    bridge = MemoryBridge(db_path=args.db_path or settings.memory_db_path)
    memos = MemoryOS(bridge, workspace_root=args.workspace)

    res = memos.ingest_clawdbot_memory_files(
        memory_dir=args.memory_dir,
        long_term_file=args.long_term_file,
        namespace=args.namespace,
    )
    print(res)
    return 0


def cmd_vector_index_jsonl(args: argparse.Namespace) -> int:
    _configure_logging()
    from memory_empire.vector_indexing.embeddings import build_default_embedders
    from memory_empire.vector_indexing.indexer import Indexer
    from memory_empire.vector_indexing.router import Router
    from memory_empire.vector_indexing.sources import FileGlobSource
    from memory_empire.vector_indexing.storage_sqlite import SQLiteIndexDB
    from memory_empire.vector_indexing.vector_store import build_vector_store

    os.makedirs(args.state_dir, exist_ok=True)
    db = SQLiteIndexDB(path=os.path.join(args.state_dir, "index.sqlite"))

    # Configure vector store via env/args
    if args.vector_store:
        os.environ["MEMORY_EMPIRE_VECTOR_STORE"] = args.vector_store
    if args.vector_dir:
        vector_dir = args.vector_dir
    else:
        vector_dir = None

    store = build_vector_store(vector_dir)
    indexer = Indexer(db=db, vector_store=store, embedders=build_default_embedders(), router=Router())

    path = args.path
    if os.path.isdir(path):
        src = FileGlobSource(path_glob=os.path.join(path, "**", "*"), source_name="files", content_type="md")
    else:
        src = FileGlobSource(path_glob=path, source_name="files", content_type="md")

    docs = list(src.iter_documents())
    stats = indexer.index_documents(docs, namespace=args.namespace)
    print(stats)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vks")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version").set_defaults(func=lambda _a: cmd_version())

    mem = sub.add_parser("memory")
    mem_sub = mem.add_subparsers(dest="mem_cmd", required=True)

    ingest = mem_sub.add_parser("ingest-clawdbot", help="Ingest memory/*.md and MEMORY.md into Memory-OS")
    ingest.add_argument("--workspace", default=str(Path.cwd()))
    ingest.add_argument("--db-path", default=None)
    ingest.add_argument("--memory-dir", default=None)
    ingest.add_argument("--long-term-file", default=None)
    ingest.add_argument("--namespace", default="personal")
    ingest.set_defaults(func=cmd_memory_ingest_clawdbot)

    vec = sub.add_parser("index")
    vec.add_argument("path", help="File or directory to index")
    vec.add_argument("--namespace", default="personal")
    vec.add_argument("--state-dir", default=".memory_empire_state")
    vec.add_argument("--vector-store", default=None, help="jsonl|chroma|qdrant")
    vec.add_argument("--vector-dir", default=None, help="Persist dir for jsonl/chroma")
    vec.set_defaults(func=cmd_vector_index_jsonl)

    return p


def app() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    app()
