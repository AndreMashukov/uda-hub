"""LangChain tool wrappers around CultPass, UDA-Hub, RAG, and memory ops."""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agentic.tools import cultpass_ops, memory, rag, udahub_ops


class UserLookupInput(BaseModel):
    user_id: str = Field(default="", description="CultPass user id if known")
    email: str = Field(default="", description="CultPass email if user id is unknown")


class UserIdInput(BaseModel):
    user_id: str = Field(description="CultPass user id")


class ExperienceQueryInput(BaseModel):
    query: str = Field(default="", description="Optional title or location filter")


class RefundInput(BaseModel):
    user_id: str
    amount_cents: int = Field(description="Refund amount in cents")
    reason: str


class TicketLookupInput(BaseModel):
    ticket_id: str = Field(default="")
    external_user_id: str = Field(default="")


class TicketUpdateInput(BaseModel):
    ticket_id: str
    status: str = Field(description="open, resolved, or escalated")
    main_issue_type: str = Field(default="")
    tags: str = Field(default="")


class SearchInput(BaseModel):
    query: str
    k: int = Field(default=3)


class RememberInput(BaseModel):
    user_id: str
    kind: str = Field(description="preference or resolution")
    text: str


class RecallInput(BaseModel):
    user_id: str
    query: str = Field(default="")
    k: int = Field(default=5)


def get_local_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            cultpass_ops.lookup_user,
            name="lookup_user",
            description="Look up a CultPass member by user_id or email.",
            args_schema=UserLookupInput,
        ),
        StructuredTool.from_function(
            cultpass_ops.get_subscription,
            name="get_subscription",
            description="Get subscription status, tier, and monthly quota for a CultPass member.",
            args_schema=UserIdInput,
        ),
        StructuredTool.from_function(
            cultpass_ops.list_reservations,
            name="list_reservations",
            description="List a member's experience reservations.",
            args_schema=UserIdInput,
        ),
        StructuredTool.from_function(
            cultpass_ops.list_experiences,
            name="list_experiences",
            description="Search the CultPass experience catalog.",
            args_schema=ExperienceQueryInput,
        ),
        StructuredTool.from_function(
            cultpass_ops.issue_refund,
            name="issue_refund",
            description="Record a refund. Use only on the escalation path after policy checks.",
            args_schema=RefundInput,
        ),
        StructuredTool.from_function(
            udahub_ops.get_ticket_bundle,
            name="get_ticket",
            description="Load a UDA-Hub ticket, metadata, and message history.",
            args_schema=TicketLookupInput,
        ),
        StructuredTool.from_function(
            udahub_ops.update_ticket,
            name="update_ticket",
            description="Update ticket status and classification on the UDA-Hub core database.",
            args_schema=TicketUpdateInput,
        ),
        StructuredTool.from_function(
            rag.retrieve_knowledge_json,
            name="search_knowledge",
            description="Retrieve the most relevant CultPass knowledge-base articles (TF-IDF RAG).",
            args_schema=SearchInput,
        ),
        StructuredTool.from_function(
            memory.remember,
            name="remember",
            description="Store a long-term memory such as a preference or past resolution.",
            args_schema=RememberInput,
        ),
        StructuredTool.from_function(
            memory.recall,
            name="recall",
            description="Recall long-term memories for a user, optionally ranked against a query.",
            args_schema=RecallInput,
        ),
    ]
