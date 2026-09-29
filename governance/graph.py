"""LangGraph StateGraph: Researcher -> Critic -> (Refiner | Approve | Fail).

Routing logic (conditional edge after critic):
  1. If pass_fail is True  -> approve -> END
  2. If pass_fail is False and iteration_count >= max_iterations -> circuit_breaker -> END
  3. Otherwise -> refiner -> critic  (loop)

This is a fail-closed governance gate: unaligned content never gets status=approved.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from governance.agents import (
    approve_node,
    circuit_breaker_node,
    critic_node,
    refiner_node,
    researcher_node,
)
from governance.config import get_settings
from governance.state import GovernanceState


def route_after_critic(state: GovernanceState) -> Literal["approve", "refiner", "circuit_breaker"]:
    """Conditional edge function. Return value MUST match add_conditional_edges mapping keys.

    Order matters: a passing draft is always approved even on the last iteration.
    Only failing drafts on the last allowed iteration trip the circuit breaker.
    """
    passed = bool(state.get("pass_fail"))
    if passed:
        return "approve"

    iteration = int(state.get("iteration_count") or 1)
    max_iterations = int(state.get("max_iterations") or get_settings().max_governance_iterations)
    if iteration >= max_iterations:
        return "circuit_breaker"

    return "refiner"


def build_governance_graph() -> StateGraph:
    graph = StateGraph(GovernanceState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("refiner", refiner_node)
    graph.add_node("approve", approve_node)
    graph.add_node("circuit_breaker", circuit_breaker_node)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "critic")

    # Critic is the only branching point in the workflow.
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "approve": "approve",
            "refiner": "refiner",
            "circuit_breaker": "circuit_breaker",
        },
    )

    # Refiner always returns to the critic; the breaker/approve nodes terminate.
    graph.add_edge("refiner", "critic")
    graph.add_edge("approve", END)
    graph.add_edge("circuit_breaker", END)
    return graph


def compile_governance_graph():
    return build_governance_graph().compile()


def initial_state(user_query: str) -> GovernanceState:
    settings = get_settings()
    return {
        "user_query": user_query.strip(),
        "draft": "",
        "critic_scores": None,
        "feedback": [],
        "iteration_count": 0,
        "pass_fail": False,
        "status": "idle",
        "active_agent": "",
        "audit_trail": [],
        "max_iterations": settings.max_governance_iterations,
        "pass_threshold": settings.pass_threshold,
        "failure_reason": "",
    }


def run_governance(user_query: str) -> GovernanceState:
    """Synchronous helper for scripts/tests. The Streamlit app prefers .stream()."""
    app = compile_governance_graph()
    return app.invoke(initial_state(user_query))
