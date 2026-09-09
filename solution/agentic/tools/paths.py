"""Absolute paths for databases and data files.

Tools and MCP servers must not rely on the process working directory.
Override with CULTPASS_DB, UDAHUB_DB, MEMORY_DB, or SOLUTION_ROOT.
"""

from __future__ import annotations

import os
from pathlib import Path


def solution_root() -> Path:
    return Path(os.environ.get("SOLUTION_ROOT", Path(__file__).resolve().parents[2]))


def cultpass_db() -> Path:
    return Path(os.environ.get("CULTPASS_DB", solution_root() / "data" / "external" / "cultpass.db"))


def udahub_db() -> Path:
    return Path(os.environ.get("UDAHUB_DB", solution_root() / "data" / "core" / "udahub.db"))


def memory_db() -> Path:
    return Path(os.environ.get("MEMORY_DB", solution_root() / "data" / "core" / "memory.db"))


def articles_jsonl() -> Path:
    return solution_root() / "data" / "external" / "cultpass_articles.jsonl"


def users_jsonl() -> Path:
    return solution_root() / "data" / "external" / "cultpass_users.jsonl"


def experiences_jsonl() -> Path:
    return solution_root() / "data" / "external" / "cultpass_experiences.jsonl"


def sqlite_url(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve()}"


# Backwards-compatible names used in a few modules
SOLUTION_ROOT = solution_root()
ARTICLES_JSONL = articles_jsonl()
