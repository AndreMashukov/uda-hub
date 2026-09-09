"""Custom LangGraph workflow for UDA-Hub.

This is not the starter create_react_agent sample and does not use
langgraph prebuilt supervisors. Nodes are wired explicitly.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agentic.agents.classifier import classifier_node
from agentic.agents.escalation import escalation_node
from agentic.agents.resolver import resolver_node
from agentic.agents.state import AgentState
from agentic.agents.supervisor import supervisor_node


def route_after_classifier(state: AgentState) -> str:
    return "escalation" if state.get("route") == "escalation" else "resolver"


def route_after_resolver(state: AgentState) -> str:
    return "escalation" if state.get("route") == "escalation" else "end"


def build_orchestrator(checkpointer: MemorySaver | None = None):
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("resolver", resolver_node)
    graph.add_node("escalation", escalation_node)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "classifier")
    graph.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {"resolver": "resolver", "escalation": "escalation"},
    )
    graph.add_conditional_edges(
        "resolver",
        route_after_resolver,
        {"escalation": "escalation", "end": END},
    )
    graph.add_edge("escalation", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


orchestrator = build_orchestrator()
