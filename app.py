"""Streamlit control plane: visualize the agentic governance monologue."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from governance.config import get_settings
from governance.constitution import ENTERPRISE_AI_CONSTITUTION
from governance.graph import compile_governance_graph, initial_state
from governance.state import GovernanceState, IterationRecord

st.set_page_config(
    page_title="Agentic AI Governance Optimizer",
    page_icon="⚖️",
    layout="wide",
)

AGENT_LABELS = {
    "researcher": "Researcher Agent (Generator)",
    "critic": "Governance Critic Agent (Evaluator / Guardrail)",
    "refiner": "Refiner Agent",
    "approve": "Governance Gate — Approved",
    "circuit_breaker": "Circuit Breaker — Governance Review Failed",
}


def _score_delta_caption(scores: Dict[str, Any] | None) -> str:
    if not scores:
        return ""
    return (
        f"Factual {scores.get('factual_accuracy_score', '—')}/10 · "
        f"Neutrality {scores.get('neutrality_bias_score', '—')}/10 · "
        f"Safety {scores.get('safety_compliance_score', '—')}/10 · "
        f"{'PASS' if scores.get('pass_fail') else 'FAIL'}"
    )


def _render_scores(scores: Dict[str, Any] | None) -> None:
    if not scores:
        st.info("No critic scores on this step.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Factual accuracy", f"{scores.get('factual_accuracy_score', '—')}/10")
    c2.metric("Neutrality", f"{scores.get('neutrality_bias_score', '—')}/10")
    c3.metric("Safety compliance", f"{scores.get('safety_compliance_score', '—')}/10")
    passed = bool(scores.get("pass_fail"))
    c4.metric("Gate", "PASS" if passed else "FAIL")
    articles = scores.get("constitution_articles_cited") or []
    if articles:
        st.caption("Articles cited: " + ", ".join(str(a) for a in articles))
    if scores.get("summary"):
        st.write(scores["summary"])


def _group_trail(trail: List[IterationRecord]) -> Dict[int, List[IterationRecord]]:
    grouped: Dict[int, List[IterationRecord]] = {}
    for record in trail:
        grouped.setdefault(int(record.get("iteration") or 1), []).append(record)
    return grouped


def render_audit_trail(trail: List[IterationRecord]) -> None:
    """Expander-per-iteration view so a recruiter can inspect the loop, not just the answer."""
    grouped = _group_trail(trail)
    for iteration in sorted(grouped):
        records = grouped[iteration]
        critic = next((r for r in records if r.get("agent") == "critic"), None)
        header_extra = _score_delta_caption(critic.get("critic_scores") if critic else None)
        title = f"Iteration {iteration}"
        if header_extra:
            title = f"{title}: {header_extra}"
        with st.expander(title, expanded=iteration == max(grouped)):
            for record in records:
                agent = record.get("agent") or "unknown"
                st.subheader(AGENT_LABELS.get(agent, agent))
                if agent in {"researcher", "refiner"}:
                    st.markdown(f"**{ 'Iteration ' + str(iteration) + ': Draft' }**")
                    st.markdown(record.get("draft") or "_(empty draft)_")
                if agent == "critic":
                    st.markdown(f"**Iteration {iteration}: Critic Scores & Feedback**")
                    _render_scores(record.get("critic_scores"))
                    feedback = record.get("feedback") or []
                    if feedback:
                        st.markdown("**Actionable feedback**")
                        for item in feedback:
                            st.markdown(f"- {item}")
                    else:
                        st.success("No deductions listed.")
                if record.get("summary") and agent != "critic":
                    st.caption(record["summary"])


def merge_update(base: GovernanceState, update: Dict[str, Any]) -> GovernanceState:
    merged = dict(base)
    merged.update(update)
    return merged  # type: ignore[return-value]


def run_stream(query: str) -> GovernanceState:
    """Stream node updates into Streamlit status + session state for live monologue."""
    graph = compile_governance_graph()
    state = initial_state(query)
    settings = get_settings()

    with st.status("Governance pipeline starting…", expanded=True) as status:
        st.write(
            f"Pass threshold ≥ {settings.pass_threshold}/10 on all axes · "
            f"Max iterations: {settings.max_governance_iterations}"
        )
        for event in graph.stream(state, stream_mode="updates"):
            for node_name, payload in event.items():
                state = merge_update(state, payload)
                label = AGENT_LABELS.get(node_name, node_name)
                status.update(label=f"Active: {label}", state="running")
                st.write(f"**{label}** finished.")
                if node_name == "critic" and payload.get("critic_scores"):
                    st.caption(_score_delta_caption(payload["critic_scores"]))
                if node_name == "circuit_breaker":
                    status.update(label="Governance Review Failed", state="error")
                if node_name == "approve":
                    status.update(label="Draft approved by governance gate", state="complete")

        if state.get("status") == "approved":
            status.update(label="Approved — output cleared for the user", state="complete")
        elif state.get("status") == "governance_review_failed":
            status.update(label="Governance Review Failed", state="error")
        else:
            status.update(label="Pipeline finished", state="complete")

    return state


def sidebar() -> None:
    with st.sidebar:
        st.header("System")
        settings = get_settings()
        st.markdown(
            f"- Provider: `Google Gemini`\n"
            f"- Model: `{settings.gemini_model}`\n"
            f"- Temperature: `{settings.gemini_temperature}`\n"
            f"- Max loops: `{settings.max_governance_iterations}`\n"
            f"- Pass threshold: `{settings.pass_threshold}`"
        )
        key_set = bool(
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or settings.resolved_api_key
        )
        st.markdown("API key: " + ("configured" if key_set else "**missing** — add `.env`"))
        with st.expander("Enterprise AI Constitution"):
            st.text(ENTERPRISE_AI_CONSTITUTION)
        st.caption(
            "Researcher drafts → Critic scores with structured output → "
            "Refiner loops until pass or circuit breaker."
        )


def main() -> None:
    sidebar()
    st.title("Agentic AI Governance Optimizer")
    st.write(
        "Multi-agent LangGraph gate that enforces Constitutional AI and RLHF-style "
        "criteria (safety, factual accuracy, neutrality) **before** a response is shown as approved."
    )

    query = st.text_area(
        "User query",
        height=140,
        placeholder=(
            "Example: Compare the leading public arguments for and against a carbon border "
            "adjustment, noting uncertainty in economic estimates."
        ),
    )
    run = st.button("Run governance pipeline", type="primary", disabled=not query.strip())

    if run:
        get_settings.cache_clear()
        try:
            final_state = run_stream(query.strip())
        except Exception as exc:  # noqa: BLE001 — surface LLM/config errors in the UI
            st.error(str(exc))
            return
        st.session_state["last_run"] = final_state

    final_state: GovernanceState | None = st.session_state.get("last_run")
    if not final_state:
        st.info("Submit a query to stream Researcher → Critic → Refiner.")
        return

    st.divider()
    st.header("Internal agentic monologue")
    render_audit_trail(list(final_state.get("audit_trail") or []))

    st.divider()
    status = final_state.get("status")
    if status == "approved":
        st.header("Approved response")
        st.success("Cleared by the Governance Critic (all scores at or above threshold).")
        st.markdown(final_state.get("draft") or "")
    elif status == "governance_review_failed":
        st.header("Governance Review Failed")
        st.error(final_state.get("failure_reason") or "Could not align the draft.")
        with st.expander("Last unapproved draft (not cleared for users)", expanded=False):
            st.markdown(final_state.get("draft") or "")
        scores = final_state.get("critic_scores")
        if scores:
            st.subheader("Final critic payload")
            _render_scores(scores)
            for item in final_state.get("feedback") or []:
                st.markdown(f"- {item}")
    else:
        st.warning(f"Unexpected terminal status: {status}")


if __name__ == "__main__":
    main()
