"""SQLite-backed translation cache.

A single sqlite file holds every ``(source_lang, target_lang, model, source_text)
-> translated_text`` pair. Its job is twofold:

* **Avoid paying DeepSeek twice** for identical strings (common in Vic3
  mods where the same country names / modifiers repeat across files).
* **Resume** an interrupted run — the next launch simply finds most texts
  already cached.

The database is thread-safe because every operation opens its own
short-lived connection (``check_same_thread=False`` + ``WAL`` mode).
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional


logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    source_lang     TEXT NOT NULL,
    target_lang     TEXT NOT NULL,
    model           TEXT NOT NULL,
    source_hash     TEXT NOT NULL,
    source_text     TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (source_lang, target_lang, model, source_hash)
);
CREATE INDEX IF NOT EXISTS idx_cache_langs
    ON cache (source_lang, target_lang, model);
"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TranslationCache:
    """Persistent translation cache."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ---------------------------------------------------------------- query
    def get_many(
        self,
        texts: Iterable[str],
        source_lang: str,
        target_lang: str,
        model: str,
    ) -> Dict[str, str]:
        """Return a ``{source_text: translated_text}`` mapping for those
        of ``texts`` that are already cached."""
        texts = list(texts)
        if not texts:
            return {}

        hashes = {_hash(t): t for t in texts}
        result: Dict[str, str] = {}

        # SQLite has a limit on the number of parameters in a single IN().
        # Chunk the query to be safe.
        chunk_size = 400
        hash_list = list(hashes.keys())
        with self._connect() as conn:
            for i in range(0, len(hash_list), chunk_size):
                chunk = hash_list[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                cursor = conn.execute(
                    f"SELECT source_hash, translated_text FROM cache "
                    f"WHERE source_lang=? AND target_lang=? AND model=? "
                    f"AND source_hash IN ({placeholders})",
                    (source_lang, target_lang, model, *chunk),
                )
                for source_hash, translated in cursor.fetchall():
                    original = hashes.get(source_hash)
                    if original is not None:
                        result[original] = translated
        return result

    def put_many(
        self,
        pairs: Iterable[tuple[str, str]],
        source_lang: str,
        target_lang: str,
        model: str,
    ) -> None:
        """Insert (or replace) a batch of ``(source, translated)`` pairs."""
        rows = []
        for source, translated in pairs:
            rows.append(
                (source_lang, target_lang, model, _hash(source), source, translated)
            )
        if not rows:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO cache "
                "(source_lang, target_lang, model, source_hash, "
                " source_text, translated_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

    # --------------------------------------------------------------- manage
    def clear(
        self,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        model: Optional[str] = None,
    ) -> int:
        """Remove cache entries. Returns the number of rows deleted.

        With no arguments, *wipes the entire cache*.
        """
        conditions = []
        params: List[str] = []
        if source_lang:
            conditions.append("source_lang=?")
            params.append(source_lang)
        if target_lang:
            conditions.append("target_lang=?")
            params.append(target_lang)
        if model:
            conditions.append("model=?")
            params.append(model)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with self._lock, self._connect() as conn:
            cur = conn.execute(f"DELETE FROM cache {where}", params)
            conn.commit()
            return cur.rowcount or 0

    def stats(self) -> Dict[str, int]:
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM cache")
            total = cur.fetchone()[0]
        return {"total": total}
