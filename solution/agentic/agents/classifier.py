"""Classifier: label the ticket, then apply deterministic routing rules."""

from __future__ import annotations

import json
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentic.agents.state import AgentState, Classification
from agentic.tools.cultpass_ops import lookup_user

CONFIDENCE_RESOLVE = 0.55

_RULES: list[tuple[str, list[str]]] = [
    ("refund", ["refund", "money back", "charged twice", "duplicate charge"]),
    ("account_blocked", ["blocked", "ban", "suspended"]),
    ("login", ["log in", "login", "password", "sign in", "can't access", "cannot access"]),
    ("billing", ["billing", "payment", "card", "charged", "invoice"]),
    ("reservation", ["reserv", "booking", "qr", "event", "experience", "venue"]),
    ("subscription", ["subscription", "cancel", "pause", "quota", "plan", "included"]),
    ("catalog", ["catalog", "what's on", "available experience", "paddleboard", "museum"]),
]


def heuristic_classify(text: str) -> Classification:
    lowered = text.lower()
    for issue_type, keywords in _RULES:
        if any(keyword in lowered for keyword in keywords):
            urgency = "high" if issue_type in {"refund", "account_blocked", "login"} else "medium"
            return Classification(
                issue_type=issue_type,  # type: ignore[arg-type]
                urgency=urgency,  # type: ignore[arg-type]
                confidence=0.82,
                needs_refund=issue_type == "refund" or "refund" in lowered,
                rationale=f"Matched keywords for {issue_type}",
            )
    return Classification(
        issue_type="general",
        urgency="low",
        confidence=0.6,
        rationale="No strong keyword match; treat as general FAQ",
    )


def _llm_classify(text: str) -> Classification | None:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VOCAREUM_API_KEY")
    if not api_key or os.environ.get("UDA_USE_HEURISTIC", "0") == "1":
        return None
    try:
        from langchain_openai import ChatOpenAI

        kwargs = {"model": os.environ.get("UDA_MODEL", "gpt-4o-mini"), "temperature": 0}
        if os.environ.get("VOCAREUM_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["VOCAREUM_API_KEY"]
            kwargs["base_url"] = os.environ.get("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
        llm = ChatOpenAI(**kwargs).with_structured_output(Classification)
        return llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Classify a CultPass customer support ticket. "
                        "issue_type must be one of: login, reservation, subscription, "
                        "billing, refund, account_blocked, catalog, general, unknown."
                    )
                ),
                HumanMessage(content=text),
            ]
        )
    except Exception:
        return None


def decide_route(classification: Classification) -> str:
    if classification.user_blocked or classification.issue_type == "account_blocked":
        return "escalation"
    if classification.needs_refund or classification.issue_type == "refund":
        return "escalation"
    if classification.confidence < CONFIDENCE_RESOLVE or classification.issue_type == "unknown":
        return "escalation"
    if classification.urgency == "high" and classification.issue_type == "login":
        # First-pass login still goes to resolver (password reset). Persistent
        # login is escalated later if the resolver remains unsure.
        return "resolver"
    return "resolver"


def classifier_node(state: AgentState) -> dict:
    context = dict(state.get("context") or {})
    text = context.get("raw_ticket") or ""
    if not text:
        for message in reversed(state.get("messages") or []):
            if isinstance(message, HumanMessage):
                text = message.content if isinstance(message.content, str) else str(message.content)
                break

    classification = _llm_classify(text) or heuristic_classify(text)

    user_id = context.get("external_user_id") or ""
    email = context.get("user_email") or ""
    if user_id or email:
        parsed = json.loads(lookup_user(user_id=user_id, email=email))
        if parsed.get("is_blocked"):
            classification.user_blocked = True
            classification.issue_type = "account_blocked"
            classification.urgency = "high"
            classification.rationale += " Member account is blocked."

    route = decide_route(classification)
    notice = AIMessage(
        content=(
            f"Classifier labeled this as {classification.issue_type} "
            f"(confidence {classification.confidence:.2f}, urgency {classification.urgency}). "
            f"Routing to {route}."
        )
    )
    return {
        "messages": [notice],
        "classification": classification.model_dump(),
        "route": route,
    }
