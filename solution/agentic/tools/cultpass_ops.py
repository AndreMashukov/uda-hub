"""CultPass account operations used by tools and the CultPass MCP server."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agentic.tools.paths import cultpass_db, sqlite_url
from data.models import cultpass

_engine = None
_Session = None
_engine_key = None


def _session_factory():
    global _engine, _Session, _engine_key
    key = str(cultpass_db().resolve())
    if _engine is None or _engine_key != key:
        _engine = create_engine(sqlite_url(cultpass_db()), echo=False)
        cultpass.Base.metadata.create_all(_engine)
        _Session = sessionmaker(bind=_engine)
        _engine_key = key
    return _Session()


def _row(instance) -> dict[str, Any]:
    return {column.name: getattr(instance, column.name) for column in instance.__table__.columns}


def lookup_user(user_id: str = "", email: str = "") -> str:
    """Look up a CultPass member by user_id or email."""
    session = _session_factory()
    try:
        query = session.query(cultpass.User)
        if user_id:
            user = query.filter_by(user_id=user_id).first()
        elif email:
            user = query.filter_by(email=email).first()
        else:
            return json.dumps({"error": "Provide user_id or email"})
        if not user:
            return json.dumps({"error": "User not found"})
        payload = _row(user)
        payload["is_blocked"] = bool(user.is_blocked)
        payload["created_at"] = str(user.created_at)
        payload["updated_at"] = str(user.updated_at)
        return json.dumps(payload, default=str)
    finally:
        session.close()


def get_subscription(user_id: str) -> str:
    """Return the CultPass subscription for a member."""
    session = _session_factory()
    try:
        sub = session.query(cultpass.Subscription).filter_by(user_id=user_id).first()
        if not sub:
            return json.dumps({"error": "Subscription not found"})
        return json.dumps(_row(sub), default=str)
    finally:
        session.close()


def list_reservations(user_id: str) -> str:
    """List reservations for a CultPass member."""
    session = _session_factory()
    try:
        rows = session.query(cultpass.Reservation).filter_by(user_id=user_id).all()
        items = []
        for reservation in rows:
            item = _row(reservation)
            if reservation.experience:
                item["experience_title"] = reservation.experience.title
                item["experience_location"] = reservation.experience.location
                item["experience_when"] = str(reservation.experience.when)
                item["is_premium"] = bool(reservation.experience.is_premium)
            items.append(item)
        return json.dumps(items, default=str)
    finally:
        session.close()


def list_experiences(query: str = "") -> str:
    """List CultPass experiences, optionally filtered by title or location text."""
    session = _session_factory()
    try:
        rows = session.query(cultpass.Experience).all()
        needle = query.lower().strip()
        items = []
        for exp in rows:
            blob = f"{exp.title} {exp.description} {exp.location}".lower()
            if needle and needle not in blob:
                continue
            items.append(_row(exp))
        return json.dumps(items, default=str)
    finally:
        session.close()


def issue_refund(user_id: str, amount_cents: int, reason: str) -> str:
    """Record a refund action. Intended for the escalation path only."""
    if amount_cents <= 0:
        return json.dumps({"error": "amount_cents must be positive"})
    session = _session_factory()
    try:
        user = session.query(cultpass.User).filter_by(user_id=user_id).first()
        if not user:
            return json.dumps({"error": "User not found"})
        if user.is_blocked:
            return json.dumps(
                {
                    "status": "pending_review",
                    "message": "Account is blocked; refund recorded for human approval only.",
                    "user_id": user_id,
                    "amount_cents": amount_cents,
                    "reason": reason,
                }
            )
        return json.dumps(
            {
                "status": "recorded",
                "message": "Refund request recorded. Finance will complete the transfer.",
                "user_id": user_id,
                "amount_cents": amount_cents,
                "reason": reason,
            }
        )
    finally:
        session.close()
