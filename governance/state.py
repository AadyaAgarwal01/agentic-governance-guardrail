"""LangGraph shared state for the governance loop."""

from typing import Any, Dict, List, Literal, Optional, TypedDict

GovernanceStatus = Literal[
    "idle",
    "researching",
    "critiquing",
    "refining",
    "approved",
    "governance_review_failed",
]


class IterationRecord(TypedDict, total=False):
    """One pass through draft (or refine) + critic, stored for the Streamlit audit UI."""

    iteration: int
    agent: str
    draft: str
    critic_scores: Dict[str, Any]
    feedback: List[str]
    pass_fail: bool
    summary: str


class GovernanceState(TypedDict, total=False):
    """Single source of truth flowing through Researcher -> Critic -> Refiner.

    LangGraph merges node return dicts into this state. We keep both the live
    fields the router needs and an append-only `audit_trail` so the UI can
    replay the agentic monologue without re-running the graph.
    """

    user_query: str
    draft: str
    critic_scores: Optional[Dict[str, Any]]
    feedback: List[str]
    iteration_count: int
    pass_fail: bool
    status: GovernanceStatus
    active_agent: str
    audit_trail: List[IterationRecord]
    max_iterations: int
    pass_threshold: int
    failure_reason: str
