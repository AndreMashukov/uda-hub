"""Seed the CultPass (external) SQLite database."""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine

from agentic.tools.paths import cultpass_db, experiences_jsonl, sqlite_url, users_jsonl
from data.models import cultpass
from utils import get_session, reset_db

SUBSCRIPTION_BY_USER = {
    "a4ab87": ("cancelled", "basic", 4),
    "f556c0": ("active", "basic", 4),
    "88382b": ("active", "premium", 8),
    "888fb2": ("active", "basic", 4),
    "f1f10d": ("cancelled", "basic", 4),
    "e6376d": ("cancelled", "premium", 8),
}


def setup_cultpass_db(db_path: str | Path | None = None) -> Path:
    path = Path(db_path) if db_path else cultpass_db()
    reset_db(str(path), echo=False)
    engine = create_engine(sqlite_url(path), echo=False)
    cultpass.Base.metadata.create_all(engine)

    with experiences_jsonl().open(encoding="utf-8") as handle:
        experience_data = [json.loads(line) for line in handle if line.strip()]
    with users_jsonl().open(encoding="utf-8") as handle:
        cultpass_users = [json.loads(line) for line in handle if line.strip()]

    rng = random.Random(42)

    with get_session(engine) as session:
        experiences = []
        for idx, experience in enumerate(experience_data):
            experiences.append(
                cultpass.Experience(
                    experience_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, experience["title"]))[:6],
                    title=experience["title"],
                    description=experience["description"],
                    location=experience["location"],
                    when=datetime.now() + timedelta(days=idx + 1),
                    slots_available=rng.randint(1, 30),
                    is_premium=(idx % 2 == 0),
                )
            )
        session.add_all(experiences)

        db_users = [
            cultpass.User(
                user_id=user_info["id"],
                full_name=user_info["name"],
                email=user_info["email"],
                is_blocked=user_info["is_blocked"],
                created_at=datetime.now(),
            )
            for user_info in cultpass_users
        ]
        session.add_all(db_users)

        subscriptions = []
        for user_info in cultpass_users:
            status, tier, quota = SUBSCRIPTION_BY_USER[user_info["id"]]
            subscriptions.append(
                cultpass.Subscription(
                    subscription_id=str(uuid.uuid5(uuid.NAMESPACE_DNS, user_info["id"]))[:6],
                    user_id=user_info["id"],
                    status=status,
                    tier=tier,
                    monthly_quota=quota,
                    started_at=datetime.now(),
                )
            )
        session.add_all(subscriptions)

        experience_ids = [exp.experience_id for exp in experiences]
        reservations = [
            cultpass.Reservation(
                reservation_id="r00001",
                user_id="a4ab87",
                experience_id=experience_ids[0],
                status="reserved",
            ),
            cultpass.Reservation(
                reservation_id="r00002",
                user_id="a4ab87",
                experience_id=experience_ids[1],
                status="reserved",
            ),
            cultpass.Reservation(
                reservation_id="r00003",
                user_id="f556c0",
                experience_id=experience_ids[2],
                status="reserved",
            ),
            cultpass.Reservation(
                reservation_id="r00004",
                user_id="88382b",
                experience_id=experience_ids[4],
                status="reserved",
            ),
            cultpass.Reservation(
                reservation_id="r00005",
                user_id="f1f10d",
                experience_id=experience_ids[3],
                status="cancelled",
            ),
        ]
        session.add_all(reservations)

    return path


if __name__ == "__main__":
    created = setup_cultpass_db()
    print(f"CultPass database ready at {created}")
