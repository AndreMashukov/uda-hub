"""UDA-Hub core operations used by tools and the UDA-Hub MCP server."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agentic.tools.paths import sqlite_url, udahub_db
from data.models import udahub

_engine = None
_Session = None
_engine_key = None


def _session_factory():
    global _engine, _Session, _engine_key
    key = str(udahub_db().resolve())
    if _engine is None or _engine_key != key:
        _engine = create_engine(sqlite_url(udahub_db()), echo=False)
        udahub.Base.metadata.create_all(_engine)
        _Session = sessionmaker(bind=_engine)
        _engine_key = key
    return _Session()


def _row(instance) -> dict[str, Any]:
    return {column.name: getattr(instance, column.name) for column in instance.__table__.columns}


def list_knowledge_articles() -> list[dict[str, Any]]:
    session = _session_factory()
    try:
        rows = session.query(udahub.Knowledge).all()
        return [
            {
                "article_id": row.article_id,
                "title": row.title,
                "content": row.content,
                "tags": row.tags or "",
            }
            for row in rows
        ]
    finally:
        session.close()


def get_ticket_bundle(ticket_id: str = "", external_user_id: str = "") -> str:
    """Load a ticket, metadata, messages, and linked UDA-Hub user."""
    session = _session_factory()
    try:
        ticket = None
        if ticket_id:
            ticket = session.query(udahub.Ticket).filter_by(ticket_id=ticket_id).first()
        elif external_user_id:
            user = (
                session.query(udahub.User)
                .filter_by(external_user_id=external_user_id, account_id="cultpass")
                .first()
            )
            if user and user.tickets:
                ticket = user.tickets[0]
        if not ticket:
            return json.dumps({"error": "Ticket not found"})
        messages = [
            {
                "role": message.role.value if hasattr(message.role, "value") else str(message.role),
                "content": message.content,
                "created_at": str(message.created_at),
            }
            for message in ticket.messages
        ]
        meta = ticket.ticket_metadata
        payload = {
            "ticket": _row(ticket),
            "metadata": _row(meta) if meta else None,
            "user": _row(ticket.user) if ticket.user else None,
            "messages": messages,
        }
        return json.dumps(payload, default=str)
    finally:
        session.close()


def update_ticket(
    ticket_id: str,
    status: str,
    main_issue_type: str = "",
    tags: str = "",
) -> str:
    """Update ticket metadata after classification, resolution, or escalation."""
    session = _session_factory()
    try:
        meta = session.query(udahub.TicketMetadata).filter_by(ticket_id=ticket_id).first()
        if not meta:
            return json.dumps({"error": "Ticket metadata not found"})
        meta.status = status
        if main_issue_type:
            meta.main_issue_type = main_issue_type
        if tags:
            meta.tags = tags
        session.commit()
        return json.dumps(_row(meta), default=str)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
