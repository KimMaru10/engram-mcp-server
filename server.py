#!/usr/bin/env python3
from __future__ import annotations

"""engram - Long-term memory MCP server for Claude Code.

Provides persistent memory across sessions using:
- SQLite FTS5 (trigram) for keyword search
- sqlite-vec + sentence-transformers (Ruri v3-310m) for vector search
- RRF (Reciprocal Rank Fusion) to merge results
- Time decay (half-life 30 days)
- Auto deduplication and pruning
"""

import json
import math
import os
import sqlite3
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Heavy imports are deferred to first use to keep MCP startup fast.
np = None
sqlite_vec = None
SentenceTransformer = None


def _lazy_import():
    """Import heavy dependencies on first use."""
    global np, sqlite_vec, SentenceTransformer
    if np is None:
        import numpy as _np
        import sqlite_vec as _sqlite_vec
        from sentence_transformers import SentenceTransformer as _ST

        np = _np
        sqlite_vec = _sqlite_vec
        SentenceTransformer = _ST

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = Path(os.environ.get("ENGRAM_DB_PATH", Path.home() / ".claude" / "engram" / "memory.db"))
MODEL_NAME = "cl-nagoya/ruri-v3-310m"
HALF_LIFE_DAYS = 30
MAX_MEMORIES = 10_000
DEDUP_THRESHOLD = 0.90
VECTOR_DIM = 768  # ruri-v3-310m output dimension
RRF_K = 60  # RRF constant

# ---------------------------------------------------------------------------
# Globals (lazy init)
# ---------------------------------------------------------------------------
_model = None  # SentenceTransformer instance (lazy loaded)
_db: sqlite3.Connection | None = None


def _get_model():
    global _model
    if _model is None:
        _lazy_import()
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _serialize_vec(vec: np.ndarray) -> bytes:
    """Serialize a float32 numpy array to bytes for sqlite-vec."""
    return vec.astype(np.float32).tobytes()


def _deserialize_vec(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is not None:
        return _db

    _lazy_import()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            content     TEXT NOT NULL,
            project     TEXT DEFAULT '',
            tags        TEXT DEFAULT '',
            created_at  REAL NOT NULL,
            last_hit_at REAL NOT NULL,
            hit_count   INTEGER DEFAULT 0,
            embedding   BLOB
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            tags,
            content='memories',
            content_rowid='id',
            tokenize='trigram'
        );

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, tags)
            VALUES (new.id, new.content, new.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags)
            VALUES ('delete', old.id, old.content, old.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags)
            VALUES ('delete', old.id, old.content, old.tags);
            INSERT INTO memories_fts(rowid, content, tags)
            VALUES (new.id, new.content, new.tags);
        END;
    """)

    # Create vec table for vector search
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[{VECTOR_DIM}]
        )
    """)

    conn.commit()
    _db = conn
    return _db


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _time_decay(created_at: float) -> float:
    """Exponential decay with configurable half-life."""
    age_days = (time.time() - created_at) / 86400
    return math.pow(0.5, age_days / HALF_LIFE_DAYS)


def _embed(text: str) -> np.ndarray:
    model = _get_model()
    return model.encode(text, normalize_embeddings=True)


def _find_duplicate(content: str, project: str) -> int | None:
    """Check if a similar memory already exists. Returns id if found."""
    db = _get_db()
    query_vec = _embed(content)

    # Search top 5 candidates by vector similarity
    rows = db.execute(
        """
        SELECT id, distance
        FROM memories_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT 5
        """,
        [_serialize_vec(query_vec)],
    ).fetchall()

    if not rows:
        return None

    for row in rows:
        mem = db.execute(
            "SELECT id, project, embedding FROM memories WHERE id = ?",
            [row["id"]],
        ).fetchone()
        if mem and mem["project"] == project and mem["embedding"]:
            sim = _cosine_similarity(query_vec, _deserialize_vec(mem["embedding"]))
            if sim >= DEDUP_THRESHOLD:
                return mem["id"]

    return None


def _save_memory(content: str, project: str = "", tags: str = "") -> dict:
    db = _get_db()
    now = time.time()
    vec = _embed(content)

    # Check for duplicate
    dup_id = _find_duplicate(content, project)
    if dup_id:
        db.execute(
            "UPDATE memories SET content = ?, tags = ?, last_hit_at = ? WHERE id = ?",
            [content, tags, now, dup_id],
        )
        db.execute(
            "DELETE FROM memories_vec WHERE id = ?", [dup_id]
        )
        db.execute(
            "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
            [dup_id, _serialize_vec(vec)],
        )
        db.commit()
        return {"status": "updated", "id": dup_id, "message": "Similar memory found and updated"}

    # Insert new
    cur = db.execute(
        "INSERT INTO memories(content, project, tags, created_at, last_hit_at, embedding) VALUES (?, ?, ?, ?, ?, ?)",
        [content, project, tags, now, now, _serialize_vec(vec)],
    )
    mem_id = cur.lastrowid
    db.execute(
        "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
        [mem_id, _serialize_vec(vec)],
    )
    db.commit()

    # Enforce max limit
    _enforce_limit(db)

    return {"status": "created", "id": mem_id, "message": "Memory saved"}


def _enforce_limit(db: sqlite3.Connection):
    count = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    if count <= MAX_MEMORIES:
        return

    excess = count - MAX_MEMORIES
    # Delete oldest with lowest hit_count first
    ids = db.execute(
        """
        SELECT id FROM memories
        ORDER BY hit_count ASC, last_hit_at ASC
        LIMIT ?
        """,
        [excess],
    ).fetchall()

    for row in ids:
        _delete_memory(db, row["id"])

    db.commit()


def _delete_memory(db: sqlite3.Connection, mem_id: int):
    db.execute("DELETE FROM memories_vec WHERE id = ?", [mem_id])
    db.execute("DELETE FROM memories WHERE id = ?", [mem_id])


def _search_fts(query: str, project: str, limit: int) -> list[tuple[int, int]]:
    """FTS5 trigram search. Returns list of (id, rank_position)."""
    db = _get_db()

    # Build query for FTS5 trigram - use raw query as trigram handles it
    sql = """
        SELECT m.id
        FROM memories_fts f
        JOIN memories m ON m.id = f.rowid
        WHERE memories_fts MATCH ?
    """
    params: list[Any] = [query]

    if project:
        sql += " AND m.project = ?"
        params.append(project)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit * 2)

    try:
        rows = db.execute(sql, params).fetchall()
        return [(row["id"], i) for i, row in enumerate(rows)]
    except Exception:
        return []


def _search_vec(query: str, project: str, limit: int) -> list[tuple[int, int]]:
    """Vector similarity search. Returns list of (id, rank_position)."""
    db = _get_db()
    query_vec = _embed(query)

    rows = db.execute(
        """
        SELECT id, distance
        FROM memories_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        [_serialize_vec(query_vec), limit * 2],
    ).fetchall()

    if not project:
        return [(row["id"], i) for i, row in enumerate(rows)]

    # Filter by project
    result = []
    rank = 0
    for row in rows:
        mem = db.execute("SELECT project FROM memories WHERE id = ?", [row["id"]]).fetchone()
        if mem and mem["project"] == project:
            result.append((row["id"], rank))
            rank += 1
    return result


def _search(query: str, project: str = "", limit: int = 10) -> list[dict]:
    """Hybrid search: FTS5 + vector with RRF fusion and time decay."""
    db = _get_db()

    fts_results = _search_fts(query, project, limit)
    vec_results = _search_vec(query, project, limit)

    # RRF fusion
    scores: dict[int, float] = {}
    for mem_id, rank in fts_results:
        scores[mem_id] = scores.get(mem_id, 0) + 1.0 / (RRF_K + rank)
    for mem_id, rank in vec_results:
        scores[mem_id] = scores.get(mem_id, 0) + 1.0 / (RRF_K + rank)

    if not scores:
        return []

    # Apply time decay and collect results
    results = []
    for mem_id, rrf_score in scores.items():
        row = db.execute(
            "SELECT id, content, project, tags, created_at, last_hit_at, hit_count FROM memories WHERE id = ?",
            [mem_id],
        ).fetchone()
        if not row:
            continue

        decay = _time_decay(row["created_at"])
        final_score = rrf_score * decay

        results.append({
            "id": row["id"],
            "content": row["content"],
            "project": row["project"],
            "tags": row["tags"],
            "score": round(final_score, 4),
            "created_at": datetime.fromtimestamp(row["created_at"], tz=timezone.utc).isoformat(),
            "hit_count": row["hit_count"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:limit]

    # Update hit counts
    now = time.time()
    for r in results:
        db.execute(
            "UPDATE memories SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
            [now, r["id"]],
        )
    db.commit()

    return results


def _prune(older_than_days: int = 90, project: str = "") -> dict:
    """Remove old, unused memories."""
    db = _get_db()
    cutoff = time.time() - (older_than_days * 86400)

    sql = "SELECT id FROM memories WHERE last_hit_at < ?"
    params: list[Any] = [cutoff]

    if project:
        sql += " AND project = ?"
        params.append(project)

    rows = db.execute(sql, params).fetchall()
    count = len(rows)

    for row in rows:
        _delete_memory(db, row["id"])

    db.commit()
    return {"deleted": count, "older_than_days": older_than_days, "project": project or "(all)"}


def _stats(project: str = "") -> dict:
    db = _get_db()

    if project:
        total = db.execute("SELECT COUNT(*) as c FROM memories WHERE project = ?", [project]).fetchone()["c"]
        oldest = db.execute("SELECT MIN(created_at) as t FROM memories WHERE project = ?", [project]).fetchone()["t"]
        newest = db.execute("SELECT MAX(created_at) as t FROM memories WHERE project = ?", [project]).fetchone()["t"]
    else:
        total = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
        oldest = db.execute("SELECT MIN(created_at) as t FROM memories").fetchone()["t"]
        newest = db.execute("SELECT MAX(created_at) as t FROM memories").fetchone()["t"]

    projects = db.execute(
        "SELECT project, COUNT(*) as c FROM memories GROUP BY project ORDER BY c DESC"
    ).fetchall()

    return {
        "total_memories": total,
        "oldest": datetime.fromtimestamp(oldest, tz=timezone.utc).isoformat() if oldest else None,
        "newest": datetime.fromtimestamp(newest, tz=timezone.utc).isoformat() if newest else None,
        "projects": [{"project": p["project"] or "(global)", "count": p["c"]} for p in projects],
        "db_size_mb": round(DB_PATH.stat().st_size / 1048576, 2) if DB_PATH.exists() else 0,
    }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP("engram")


@mcp.tool()
def save(content: str, project: str = "", tags: str = "") -> str:
    """Save a memory.

    Args:
        content: The text content to remember.
        project: Project identifier (e.g. working directory path). Leave empty for global memories.
        tags: Comma-separated tags for categorization (e.g. "decision,architecture").
    """
    result = _save_memory(content, project, tags)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def search(query: str, project: str = "", limit: int = 5) -> str:
    """Search memories using hybrid keyword + semantic search.

    Args:
        query: Search query (natural language or keywords).
        project: Filter by project. Leave empty to search all.
        limit: Max number of results (default 5).
    """
    results = _search(query, project, limit)
    if not results:
        return json.dumps({"results": [], "message": "No memories found"}, ensure_ascii=False)
    return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False, indent=2)


@mcp.tool()
def prune(older_than_days: int = 90, project: str = "") -> str:
    """Remove old, unused memories.

    Args:
        older_than_days: Delete memories not accessed in this many days (default 90).
        project: Only prune this project. Leave empty for all.
    """
    result = _prune(older_than_days, project)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def stats(project: str = "") -> str:
    """Show memory statistics.

    Args:
        project: Filter stats by project. Leave empty for overall stats.
    """
    result = _stats(project)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def delete(memory_id: int) -> str:
    """Delete a specific memory by ID.

    Args:
        memory_id: The ID of the memory to delete.
    """
    db = _get_db()
    row = db.execute("SELECT id FROM memories WHERE id = ?", [memory_id]).fetchone()
    if not row:
        return json.dumps({"status": "error", "message": f"Memory {memory_id} not found"})
    _delete_memory(db, memory_id)
    db.commit()
    return json.dumps({"status": "deleted", "id": memory_id})


if __name__ == "__main__":
    mcp.run(transport="stdio")
