"""MCP server exposing UDA-Hub core, RAG, and memory tools over stdio."""

from __future__ import annotations

import sys
from pathlib import Path

from fastmcp import FastMCP

SOLUTION_ROOT = Path(__file__).resolve().parents[3]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from agentic.tools import memory, rag, udahub_ops  # noqa: E402

mcp = FastMCP("udahub")


@mcp.tool()
def get_ticket(ticket_id: str = "", external_user_id: str = "") -> str:
    """Load a UDA-Hub ticket bundle."""
    return udahub_ops.get_ticket_bundle(ticket_id=ticket_id, external_user_id=external_user_id)


@mcp.tool()
def update_ticket(ticket_id: str, status: str, main_issue_type: str = "", tags: str = "") -> str:
    """Update ticket metadata."""
    return udahub_ops.update_ticket(
        ticket_id=ticket_id,
        status=status,
        main_issue_type=main_issue_type,
        tags=tags,
    )


@mcp.tool()
def search_knowledge(query: str, k: int = 3) -> str:
    """Retrieve knowledge-base articles with TF-IDF RAG."""
    return rag.retrieve_knowledge_json(query, k=k)


@mcp.tool()
def remember(user_id: str, kind: str, text: str) -> str:
    """Store long-term memory."""
    return memory.remember(user_id, kind, text)


@mcp.tool()
def recall(user_id: str, query: str = "", k: int = 5) -> str:
    """Recall long-term memories."""
    return memory.recall(user_id, query, k)


if __name__ == "__main__":
    mcp.run()
