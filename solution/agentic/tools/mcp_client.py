"""Load tools from local FastMCP servers, with a local-function fallback."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from agentic.tools.langchain_tools import get_local_tools
from agentic.tools.paths import solution_root

logger = logging.getLogger(__name__)


def _server_path(name: str) -> str:
    return str(solution_root() / "agentic" / "tools" / "mcp" / name)


def load_mcp_tools() -> list[Any]:
    """Start CultPass and UDA-Hub MCP servers over stdio and return LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "cultpass": {
                "command": sys.executable,
                "args": [_server_path("cultpass_server.py")],
                "transport": "stdio",
                "env": {**os.environ, "SOLUTION_ROOT": str(solution_root())},
            },
            "udahub": {
                "command": sys.executable,
                "args": [_server_path("udahub_server.py")],
                "transport": "stdio",
                "env": {**os.environ, "SOLUTION_ROOT": str(solution_root())},
            },
        }
    )
    return asyncio.run(client.get_tools())


def get_tools(use_mcp: bool | None = None) -> list[Any]:
    if use_mcp is None:
        use_mcp = os.environ.get("USE_MCP", "0") == "1"
    if not use_mcp:
        return get_local_tools()
    try:
        tools = load_mcp_tools()
        if tools:
            logger.info("Loaded %s tools from MCP servers", len(tools))
            return tools
    except Exception as exc:
        logger.warning("MCP tool load failed (%s); using local tools", exc)
    return get_local_tools()
