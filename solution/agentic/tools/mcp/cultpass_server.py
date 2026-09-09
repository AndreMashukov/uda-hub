"""MCP server exposing CultPass database tools over stdio."""

from __future__ import annotations

import sys
from pathlib import Path

from fastmcp import FastMCP

SOLUTION_ROOT = Path(__file__).resolve().parents[3]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from agentic.tools import cultpass_ops  # noqa: E402

mcp = FastMCP("cultpass")


@mcp.tool()
def lookup_user(user_id: str = "", email: str = "") -> str:
    """Look up a CultPass member by user_id or email."""
    return cultpass_ops.lookup_user(user_id=user_id, email=email)


@mcp.tool()
def get_subscription(user_id: str) -> str:
    """Get subscription status, tier, and quota."""
    return cultpass_ops.get_subscription(user_id)


@mcp.tool()
def list_reservations(user_id: str) -> str:
    """List reservations for a member."""
    return cultpass_ops.list_reservations(user_id)


@mcp.tool()
def list_experiences(query: str = "") -> str:
    """Search experiences by title or location."""
    return cultpass_ops.list_experiences(query)


@mcp.tool()
def issue_refund(user_id: str, amount_cents: int, reason: str) -> str:
    """Record a refund request."""
    return cultpass_ops.issue_refund(user_id, amount_cents, reason)


if __name__ == "__main__":
    mcp.run()
