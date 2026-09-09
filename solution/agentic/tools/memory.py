"""Long-term memory store with the same lexical scorer used for RAG.

Short-term (session) memory is LangGraph thread_id + MemorySaver.
Long-term memory stores preferences and past resolutions per user.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from agentic.tools.paths import memory_db
from agentic.tools.rag import tokenize


def _connect() -> sqlite3.Connection:
    path = memory_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def remember(user_id: str, kind: str, text: str) -> str:
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO memories (user_id, kind, text) VALUES (?, ?, ?)",
            (user_id, kind, text),
        )
        conn.commit()
        return json.dumps({"status": "saved", "user_id": user_id, "kind": kind})
    finally:
        conn.close()


def recall(user_id: str, query: str = "", k: int = 5) -> str:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, kind, text, created_at FROM memories WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
        items: list[dict[str, Any]] = [
            {"id": row[0], "kind": row[1], "text": row[2], "created_at": row[3]}
            for row in rows
        ]
        if not query or not items:
            return json.dumps(items[:k])
        query_tokens = set(tokenize(query))
        scored = []
        for item in items:
            overlap = len(query_tokens & set(tokenize(item["text"])))
            scored.append((overlap, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return json.dumps([item for overlap, item in scored[:k] if overlap > 0 or not query_tokens])
    finally:
        conn.close()
