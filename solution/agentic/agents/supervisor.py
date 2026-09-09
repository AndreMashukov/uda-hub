"""Supervisor: intake ticket metadata, load context, hand off to the classifier."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agentic.agents.state import AgentState, TicketContext
from agentic.tools.cultpass_ops import lookup_user
from agentic.tools.udahub_ops import get_ticket_bundle


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_USER_ID = re.compile(r"\b[a-f0-9]{6}\b")


def _latest_human(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def _extract_ids(text: str) -> tuple[str, str]:
    email_match = _EMAIL.search(text)
    email = email_match.group(0) if email_match else ""
    user_id = ""
    id_match = re.search(r"user[_ ]id[:\s]+([a-f0-9]{6})", text, re.I)
    if id_match:
        user_id = id_match.group(1)
    elif _USER_ID.search(text) and "ticket" not in text.lower():
        user_id = _USER_ID.search(text).group(0)
    return user_id, email


def supervisor_node(state: AgentState) -> dict:
    messages = list(state.get("messages") or [])
    existing = dict(state.get("context") or {})
    text = _latest_human(messages)
    context = TicketContext.model_validate(existing) if existing else TicketContext()
    context.raw_ticket = text or context.raw_ticket

    user_id, email = _extract_ids(text)
    if user_id:
        context.external_user_id = user_id
    if email:
        context.user_email = email

    notes = []
    if context.external_user_id or context.user_email:
        lookup = lookup_user(user_id=context.external_user_id, email=context.user_email)
        notes.append(f"lookup_user: {lookup}")
        parsed = json.loads(lookup)
        if "error" not in parsed:
            context.external_user_id = parsed.get("user_id", context.external_user_id)
            context.user_email = parsed.get("email", context.user_email)
            context.user_name = parsed.get("full_name", context.user_name)

    if context.external_user_id or context.ticket_id:
        bundle = get_ticket_bundle(
            ticket_id=context.ticket_id,
            external_user_id=context.external_user_id,
        )
        notes.append(f"get_ticket: {bundle[:500]}")
        parsed = json.loads(bundle)
        if "error" not in parsed:
            ticket = parsed.get("ticket") or {}
            user = parsed.get("user") or {}
            context.ticket_id = ticket.get("ticket_id", context.ticket_id)
            context.channel = ticket.get("channel") or context.channel
            context.external_user_id = user.get("external_user_id", context.external_user_id)
            context.user_name = user.get("user_name", context.user_name)
            history = parsed.get("messages") or []
            if history and not context.raw_ticket:
                context.raw_ticket = history[0].get("content") or ""

    ack = AIMessage(
        content=(
            "Supervisor received the ticket, loaded CultPass/UDA-Hub context, "
            "and is routing it to the classifier."
        )
    )
    return {
        "messages": [ack],
        "context": context.model_dump(),
        "tool_notes": notes,
    }
