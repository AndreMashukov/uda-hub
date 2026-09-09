"""Resolver: retrieve knowledge, call account tools, answer or send back for escalation."""

from __future__ import annotations

import json
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentic.agents.state import AgentState, Classification
from agentic.tools.cultpass_ops import get_subscription, list_experiences, list_reservations, lookup_user
from agentic.tools.memory import recall, remember
from agentic.tools.rag import retrieve_knowledge
from agentic.tools.udahub_ops import update_ticket

RESOLVE_THRESHOLD = 0.62


def _draft_from_articles(text: str, articles: list[dict], extras: list[str]) -> tuple[str, float]:
    if not articles:
        return (
            "I could not find a matching knowledge article. A specialist should review this ticket.",
            0.2,
        )
    top = articles[0]
    phrasing = ""
    if "**Suggested phrasing:**" in top["content"]:
        phrasing = top["content"].split("**Suggested phrasing:**", 1)[1].strip().strip('"')
    body = phrasing or top["content"][:600]
    extra_block = "\n".join(extras)
    reply = body
    if extra_block:
        reply = f"{body}\n\nAccount details I checked:\n{extra_block}"
    reply += f"\n\n(Source: {top['title']})"
    confidence = min(0.93, 0.55 + float(top.get("score") or 0) * 0.5)
    if "escalate to human" in top["content"].lower() and "persistent" in text.lower():
        confidence = 0.4
    return reply, confidence


def _llm_polish(ticket: str, draft: str, articles: list[dict]) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VOCAREUM_API_KEY")
    if not api_key or os.environ.get("UDA_USE_HEURISTIC", "0") == "1":
        return None
    try:
        from langchain_openai import ChatOpenAI

        kwargs = {"model": os.environ.get("UDA_MODEL", "gpt-4o-mini"), "temperature": 0.2}
        if os.environ.get("VOCAREUM_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["VOCAREUM_API_KEY"]
            kwargs["base_url"] = os.environ.get("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
        llm = ChatOpenAI(**kwargs)
        sources = "\n\n".join(f"#{i+1} {a['title']}\n{a['content']}" for i, a in enumerate(articles[:3]))
        result = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the CultPass resolver agent. Answer using the retrieved articles. "
                        "If the articles are insufficient, say so. Do not invent refunds or unblocks."
                    )
                ),
                HumanMessage(content=f"Ticket:\n{ticket}\n\nSources:\n{sources}\n\nDraft:\n{draft}"),
            ]
        )
        return result.content if isinstance(result.content, str) else str(result.content)
    except Exception:
        return None


def resolver_node(state: AgentState) -> dict:
    context = dict(state.get("context") or {})
    classification = Classification.model_validate(state.get("classification") or {})
    text = context.get("raw_ticket") or ""
    user_id = context.get("external_user_id") or ""
    extras: list[str] = []

    if user_id:
        extras.append(lookup_user(user_id=user_id))
        extras.append(get_subscription(user_id))
        extras.append(list_reservations(user_id))
        extras.append(recall(user_id, query=text, k=3))
    if classification.issue_type == "catalog":
        extras.append(list_experiences(query=text))

    articles = retrieve_knowledge(text, k=3)
    draft, confidence = _draft_from_articles(text, articles, extras[:4])
    polished = _llm_polish(text, draft, articles)
    reply = polished or draft

    if confidence < RESOLVE_THRESHOLD:
        return {
            "messages": [
                AIMessage(
                    content="Resolver confidence is too low; handing this ticket to escalation."
                )
            ],
            "retrieved": articles,
            "resolution": reply,
            "route": "escalation",
        }

    ticket_id = context.get("ticket_id") or ""
    if ticket_id:
        update_ticket(
            ticket_id=ticket_id,
            status="resolved",
            main_issue_type=classification.issue_type,
            tags=classification.issue_type,
        )
    if user_id:
        remember(user_id, "resolution", f"{classification.issue_type}: {reply[:400]}")

    return {
        "messages": [AIMessage(content=reply)],
        "retrieved": articles,
        "resolution": reply,
        "route": "end",
        "classification": {**classification.model_dump(), "confidence": confidence},
    }
