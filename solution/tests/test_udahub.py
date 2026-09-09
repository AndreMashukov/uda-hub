from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from agentic.agents.classifier import decide_route, heuristic_classify
from agentic.tools.cultpass_ops import get_subscription, lookup_user
from agentic.tools.memory import recall, remember
from agentic.tools.rag import retrieve_knowledge
from agentic.workflow import build_orchestrator
from setup_core_db import load_articles, setup_udahub_db
from setup_external_db import setup_cultpass_db


@pytest.fixture()
def seeded_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLUTION_ROOT", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("CULTPASS_DB", str(tmp_path / "cultpass.db"))
    monkeypatch.setenv("UDAHUB_DB", str(tmp_path / "udahub.db"))
    monkeypatch.setenv("MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("UDA_USE_HEURISTIC", "1")
    setup_cultpass_db(tmp_path / "cultpass.db")
    setup_udahub_db(tmp_path / "udahub.db")
    return tmp_path


def test_at_least_fourteen_articles():
    assert len(load_articles()) >= 14


def test_lookup_user_and_subscription(seeded_env):
    user = lookup_user(user_id="f556c0")
    assert "Bob Stone" in user
    assert "false" in user.lower() or '"is_blocked": false' in user.replace("False", "false")
    sub = get_subscription("f556c0")
    assert "active" in sub
    blocked = lookup_user(email="alice.kingsley@wonderland.com")
    assert "true" in blocked.lower() or "True" in blocked


def test_rag_login_ranks_login_article(seeded_env):
    hits = retrieve_knowledge("I cannot log in and need a password reset", k=3)
    assert hits
    titles = " ".join(hit["title"].lower() for hit in hits)
    assert "login" in titles


def test_rag_reservation_ranks_reservation_article(seeded_env):
    hits = retrieve_knowledge("How do I reserve a spot for an event?", k=3)
    assert hits[0]["title"].lower().find("reserve") >= 0


def test_classifier_routes():
    login = heuristic_classify("I can't log in to my CultPass account")
    assert login.issue_type == "login"
    assert decide_route(login) == "resolver"

    refund = heuristic_classify("I want a refund for a duplicate charge")
    assert refund.needs_refund or refund.issue_type == "refund"
    assert decide_route(refund) == "escalation"

    blocked = heuristic_classify("why is my account blocked")
    blocked.user_blocked = True
    assert decide_route(blocked) == "escalation"


def test_graph_resolves_reservation_question(seeded_env):
    graph = build_orchestrator()
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="user_id f556c0: How do I reserve a spot for a CultPass experience?"
                )
            ]
        },
        config={"configurable": {"thread_id": "t-reserve"}},
    )
    assert result["route"] == "end"
    assert result["classification"]["issue_type"] == "reservation"
    text = result["messages"][-1].content.lower()
    assert "reserve" in text


def test_graph_escalates_blocked_user(seeded_env):
    graph = build_orchestrator()
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="user_id a4ab87: I can't log in to my Cultpass account.")
            ]
        },
        config={"configurable": {"thread_id": "t-blocked"}},
    )
    assert result["route"] == "end"
    assert "escalat" in result["messages"][-1].content.lower()
    assert result["classification"]["user_blocked"] is True


def test_graph_escalates_refund(seeded_env):
    graph = build_orchestrator()
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="user_id f556c0: I want a refund, you charged me twice")
            ]
        },
        config={"configurable": {"thread_id": "t-refund"}},
    )
    assert "refund" in result["messages"][-1].content.lower()
    assert "escalat" in result["messages"][-1].content.lower()


def test_long_term_memory_roundtrip(seeded_env):
    remember("f556c0", "preference", "Prefers museum experiences in Sao Paulo")
    payload = recall("f556c0", query="museum", k=3)
    assert "Sao Paulo" in payload or "museum" in payload.lower()


def test_short_term_memory_uses_thread_id(seeded_env):
    graph = build_orchestrator()
    config = {"configurable": {"thread_id": "same-thread"}}
    graph.invoke(
        {"messages": [HumanMessage(content="user_id f556c0: What is included in my subscription?")]},
        config=config,
    )
    state = graph.get_state(config)
    assert state.values.get("messages")
    assert len(state.values["messages"]) >= 2
