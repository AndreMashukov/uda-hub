from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from pydantic import BaseModel, Field


IssueType = Literal[
    "login",
    "reservation",
    "subscription",
    "billing",
    "refund",
    "account_blocked",
    "catalog",
    "general",
    "unknown",
]


class Classification(BaseModel):
    issue_type: IssueType = "unknown"
    urgency: Literal["low", "medium", "high"] = "medium"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    needs_refund: bool = False
    user_blocked: bool = False
    rationale: str = ""


class TicketContext(BaseModel):
    ticket_id: str = ""
    account_id: str = "cultpass"
    channel: str = "chat"
    external_user_id: str = ""
    user_email: str = ""
    user_name: str = ""
    raw_ticket: str = ""


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    context: dict[str, Any]
    classification: dict[str, Any]
    route: str
    retrieved: list[dict[str, Any]]
    resolution: str
    escalation_summary: str
    tool_notes: list[str]
