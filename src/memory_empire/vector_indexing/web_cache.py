from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import urllib.request
from dataclasses import dataclass

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS web_cache (
  url TEXT PRIMARY KEY,
  fetched_at REAL NOT NULL,
  status INTEGER,
  etag TEXT,
  last_modified TEXT,
  content_sha256 TEXT,
  content_path TEXT
);
"""


def _safe_filename(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return h


@dataclass
class WebCache:
    """A persistent web cache inspired by claude-research-team patterns.

    It stores the raw response body on disk and metadata in sqlite.
    Uses conditional requests (ETag / Last-Modified) when possible.
    """

    db_path: str
    body_dir: str
    user_agent: str = "vector-indexing-bot/1.0"

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def init(self) -> None:
        os.makedirs(self.body_dir, exist_ok=True)
        con = self.connect()
        try:
            con.executescript(SCHEMA)
            con.commit()
        finally:
            con.close()

    def get_meta(self, url: str) -> dict | None:
        con = self.connect()
        try:
            row = con.execute(
                "SELECT url,fetched_at,status,etag,last_modified,content_sha256,content_path FROM web_cache WHERE url=?",
                (url,),
            ).fetchone()
            if not row:
                return None
            return {
                "url": row[0],
                "fetched_at": row[1],
                "status": row[2],
                "etag": row[3],
                "last_modified": row[4],
                "content_sha256": row[5],
                "content_path": row[6],
            }
        finally:
            con.close()

    def fetch(self, url: str, *, max_bytes: int = 8_000_000, timeout: int = 30) -> dict:
        self.init()
        meta = self.get_meta(url)

        headers = {"User-Agent": self.user_agent}
        if meta:
            if meta.get("etag"):
                headers["If-None-Match"] = meta["etag"]
            if meta.get("last_modified"):
                headers["If-Modified-Since"] = meta["last_modified"]

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                body = resp.read(max_bytes + 1)
                if len(body) > max_bytes:
                    body = body[:max_bytes]
                etag = resp.headers.get("ETag")
                last_modified = resp.headers.get("Last-Modified")

                sha = hashlib.sha256(body).hexdigest()
                path = os.path.join(self.body_dir, _safe_filename(url))
                with open(path, "wb") as f:
                    f.write(body)

                con = self.connect()
                try:
                    con.execute(
                        """
                        INSERT OR REPLACE INTO web_cache(url,fetched_at,status,etag,last_modified,content_sha256,content_path)
                        VALUES(?,?,?,?,?,?,?)
                        """,
                        (url, time.time(), status, etag, last_modified, sha, path),
                    )
                    con.commit()
                finally:
                    con.close()

                return {
                    "url": url,
                    "status": status,
                    "etag": etag,
                    "last_modified": last_modified,
                    "content_sha256": sha,
                    "content_path": path,
                    "body_bytes": len(body),
                    "from_cache": False,
                }
        except urllib.error.HTTPError as e:
            if e.code == 304 and meta and meta.get("content_path"):
                return {
                    "url": url,
                    "status": 304,
                    "etag": meta.get("etag"),
                    "last_modified": meta.get("last_modified"),
                    "content_sha256": meta.get("content_sha256"),
                    "content_path": meta.get("content_path"),
                    "body_bytes": None,
                    "from_cache": True,
                }
            raise

    def load_text(self, url: str, *, encoding: str = "utf-8") -> str:
        meta = self.get_meta(url)
        if not meta or not meta.get("content_path"):
            raise FileNotFoundError(f"No cached body for {url}")
        with open(meta["content_path"], "rb") as f:
            b = f.read()
        return b.decode(encoding, errors="replace")
