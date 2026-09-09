"""Escalation agent: summarize, optionally record a refund, mark the ticket."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from agentic.agents.state import AgentState, Classification
from agentic.tools.cultpass_ops import get_subscription, issue_refund, lookup_user
from agentic.tools.memory import remember
from agentic.tools.udahub_ops import update_ticket


def escalation_node(state: AgentState) -> dict:
    context = dict(state.get("context") or {})
    classification = Classification.model_validate(state.get("classification") or {})
    user_id = context.get("external_user_id") or ""
    ticket_text = context.get("raw_ticket") or ""
    notes = []

    if user_id:
        notes.append(lookup_user(user_id=user_id))
        notes.append(get_subscription(user_id))

    refund_line = ""
    if classification.needs_refund or classification.issue_type == "refund":
        if user_id:
            refund_line = issue_refund(
                user_id=user_id,
                amount_cents=1000,
                reason=ticket_text[:200] or "customer refund request",
            )
            notes.append(refund_line)
        else:
            refund_line = json.dumps({"status": "pending_review", "message": "Refund needs a user id"})

    reason = classification.rationale or "Policy requires a human specialist."
    if classification.user_blocked:
        reason = "Account is blocked; automatic resolution is not allowed."

    summary = (
        "Escalated to human support.\n"
        f"- Issue type: {classification.issue_type}\n"
        f"- Urgency: {classification.urgency}\n"
        f"- Confidence: {classification.confidence:.2f}\n"
        f"- Member: {context.get('user_name') or user_id or 'unknown'}\n"
        f"- Reason: {reason}\n"
        f"- Original ticket: {ticket_text}\n"
    )
    if refund_line:
        summary += f"- Refund tool: {refund_line}\n"
    summary += (
        "A specialist will continue from this summary. "
        "Human hours: 09:00-18:00 America/Sao_Paulo, Monday to Saturday."
    )

    ticket_id = context.get("ticket_id") or ""
    if ticket_id:
        update_ticket(
            ticket_id=ticket_id,
            status="escalated",
            main_issue_type=classification.issue_type,
            tags=f"escalated,{classification.issue_type}",
        )
    if user_id:
        remember(user_id, "escalation", summary[:500])

    return {
        "messages": [AIMessage(content=summary)],
        "escalation_summary": summary,
        "resolution": summary,
        "route": "end",
        "tool_notes": notes,
    }
