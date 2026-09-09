"""Seed the UDA-Hub core SQLite database, including CultPass knowledge articles."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from sqlalchemy import create_engine

from agentic.tools.paths import articles_jsonl, sqlite_url, udahub_db, users_jsonl
from data.models import udahub
from utils import get_session, reset_db

ACCOUNT_ID = "cultpass"
ACCOUNT_NAME = "CultPass Card"


def load_articles() -> list[dict]:
    articles = []
    with articles_jsonl().open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                articles.append(json.loads(line))
    if len(articles) < 14:
        raise AssertionError("You should load the articles with at least 14 records")
    return articles


def setup_udahub_db(db_path: str | Path | None = None) -> Path:
    path = Path(db_path) if db_path else udahub_db()
    reset_db(str(path), echo=False)
    engine = create_engine(sqlite_url(path), echo=False)
    udahub.Base.metadata.create_all(bind=engine)

    articles = load_articles()
    with users_jsonl().open(encoding="utf-8") as handle:
        cultpass_users = [json.loads(line) for line in handle if line.strip()]

    with get_session(engine) as session:
        session.add(udahub.Account(account_id=ACCOUNT_ID, account_name=ACCOUNT_NAME))
        kb = [
            udahub.Knowledge(
                article_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, article["title"])),
                account_id=ACCOUNT_ID,
                title=article["title"],
                content=article["content"],
                tags=article["tags"],
            )
            for article in articles
        ]
        session.add_all(kb)

        owner = cultpass_users[0]
        user = udahub.User(
            user_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, owner["id"])),
            account_id=ACCOUNT_ID,
            external_user_id=owner["id"],
            user_name=owner["name"],
        )
        ticket = udahub.Ticket(
            ticket_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, "alice-login")),
            account_id=ACCOUNT_ID,
            user_id=user.user_id,
            channel="chat",
        )
        metadata = udahub.TicketMetadata(
            ticket_id=ticket.ticket_id,
            status="open",
            main_issue_type=None,
            tags="login, access",
        )
        first_message = udahub.TicketMessage(
            message_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, "alice-login-msg")),
            ticket_id=ticket.ticket_id,
            role=udahub.RoleEnum.user,
            content="I can't log in to my Cultpass account.",
        )
        session.add_all([user, ticket, metadata, first_message])

        bob = next(u for u in cultpass_users if u["id"] == "f556c0")
        bob_user = udahub.User(
            user_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, bob["id"])),
            account_id=ACCOUNT_ID,
            external_user_id=bob["id"],
            user_name=bob["name"],
        )
        session.add(bob_user)

    return path


if __name__ == "__main__":
    created = setup_udahub_db()
    print(f"UDA-Hub database ready at {created}")
