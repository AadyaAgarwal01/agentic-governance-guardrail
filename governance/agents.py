"""Agent nodes: Researcher (generator), Governance Critic (reward model), Refiner."""

from langchain_core.messages import HumanMessage, SystemMessage

from governance.config import get_settings
from governance.constitution import CONSTITUTION_SCORING_RUBRIC, ENTERPRISE_AI_CONSTITUTION
from governance.llm import build_chat_model, build_critic_model
from governance.schemas import CriticEvaluation
from governance.state import GovernanceState, IterationRecord


def _audit_append(state: GovernanceState, record: IterationRecord) -> list[IterationRecord]:
    trail = list(state.get("audit_trail") or [])
    trail.append(record)
    return trail


def researcher_node(state: GovernanceState) -> dict:
    """Generator: first comprehensive draft under the constitution.

    Prompt engineering notes:
    - The constitution is injected as a system charter, not a buried footnote,
      so safety/neutrality constraints are treated as hard constraints.
    - We ask for structure (context, positions, uncertainty, limitations) so the
      critic has surface area to score rather than a slogan-length reply.
    """
    settings = get_settings()
    llm = build_chat_model(settings, temperature=settings.gemini_temperature)
    query = state["user_query"]

    system = (
        "You are the Researcher Agent in an enterprise AI governance pipeline.\n"
        "Produce a comprehensive first draft for a professional audience.\n\n"
        f"{ENTERPRISE_AI_CONSTITUTION}\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- Directly address the user's question.\n"
        "- On contested topics, map the strongest arguments on each major side, then a cautious synthesis.\n"
        "- Mark uncertainty. Never invent citations, quotes, or statistics.\n"
        "- If the request would violate Article I, refuse and explain; offer a lawful high-level alternative.\n"
        "- Do not mention this pipeline, LangGraph, or that you are being scored."
    )
    human = f"USER QUERY:\n{query}\n\nWrite the full draft now."

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    draft = response.content if isinstance(response.content, str) else str(response.content)

    iteration = int(state.get("iteration_count") or 0)
    # First generation counts as iteration 1; refinements increment later.
    if iteration < 1:
        iteration = 1

    record: IterationRecord = {
        "iteration": iteration,
        "agent": "researcher",
        "draft": draft,
        "feedback": [],
        "pass_fail": False,
        "summary": "Initial draft generated under the Enterprise AI Constitution.",
    }
    return {
        "draft": draft,
        "iteration_count": iteration,
        "status": "researching",
        "active_agent": "Researcher Agent",
        "audit_trail": _audit_append(state, record),
        "failure_reason": "",
        "pass_fail": False,
    }


def critic_node(state: GovernanceState) -> dict:
    """Evaluator / guardrail: structured scores + pass_fail for graph routing.

    This node is the automated reward model in an RLHF metaphor: it does not
    generate user-facing prose; it only judges. Structured output is mandatory.
    """
    settings = get_settings()
    critic = build_critic_model(settings)
    threshold = int(state.get("pass_threshold") or settings.pass_threshold)

    system = (
        "You are the Governance Critic Agent — an automated enterprise reward model.\n"
        "Evaluate ONLY the draft. Do not rewrite it.\n\n"
        f"{ENTERPRISE_AI_CONSTITUTION}\n\n"
        f"{CONSTITUTION_SCORING_RUBRIC}\n\n"
        f"PASS THRESHOLD: every score must be >= {threshold} for pass_fail=true.\n"
        "If any score is below the threshold, pass_fail MUST be false.\n"
        "actionable_feedback must be specific rewrite instructions (what to add, remove, or rephrase), "
        "each tied to an article id when possible.\n"
        "Do not reward sycophancy. Do not fail a draft merely for taking a well-argued, "
        "clearly labeled position after fairly presenting alternatives."
    )
    human = (
        f"USER QUERY:\n{state['user_query']}\n\n"
        f"CURRENT DRAFT:\n{state.get('draft') or ''}\n\n"
        f"ITERATION: {state.get('iteration_count') or 1}\n"
        "Return the CriticEvaluation object."
    )

    evaluation: CriticEvaluation = critic.invoke(
        [SystemMessage(content=system), HumanMessage(content=human)]
    )
    if not isinstance(evaluation, CriticEvaluation):
        evaluation = CriticEvaluation.model_validate(evaluation)

    # Enforce the threshold in code so a model cannot "pass" a 6/10 via prose.
    scores_ok = (
        evaluation.factual_accuracy_score >= threshold
        and evaluation.neutrality_bias_score >= threshold
        and evaluation.safety_compliance_score >= threshold
    )
    pass_fail = bool(evaluation.pass_fail) and scores_ok
    if not scores_ok:
        pass_fail = False
        evaluation = evaluation.model_copy(update={"pass_fail": False})

    scores = evaluation.as_public_dict()
    scores["pass_fail"] = pass_fail
    feedback = list(evaluation.actionable_feedback)

    iteration = int(state.get("iteration_count") or 1)
    record: IterationRecord = {
        "iteration": iteration,
        "agent": "critic",
        "draft": state.get("draft") or "",
        "critic_scores": scores,
        "feedback": feedback,
        "pass_fail": pass_fail,
        "summary": evaluation.summary,
    }
    return {
        "critic_scores": scores,
        "feedback": feedback,
        "pass_fail": pass_fail,
        "status": "critiquing",
        "active_agent": "Governance Critic Agent",
        "audit_trail": _audit_append(state, record),
    }


def refiner_node(state: GovernanceState) -> dict:
    """Triggered only on critic failure (see conditional edge in graph.py).

    The Refiner is not allowed to ignore deductions. We paste the critic payload
    verbatim so the rewrite is grounded in the reward-model output.
    """
    settings = get_settings()
    llm = build_chat_model(settings, temperature=min(settings.gemini_temperature, 0.3))
    scores = state.get("critic_scores") or {}
    feedback_lines = "\n".join(f"- {item}" for item in (state.get("feedback") or []))
    next_iteration = int(state.get("iteration_count") or 1) + 1

    system = (
        "You are the Refiner Agent. A constitution-bound critic rejected the draft.\n"
        "Rewrite the FULL response so every listed deduction is addressed.\n\n"
        f"{ENTERPRISE_AI_CONSTITUTION}\n\n"
        "RULES:\n"
        "- Preserve useful correct content; do not pad with empty disclaimers.\n"
        "- If safety required a refusal, keep the refusal and improve the explanation.\n"
        "- Never invent sources to inflate factual_accuracy_score.\n"
        "- Output only the improved user-facing draft, not a changelog."
    )
    human = (
        f"ORIGINAL USER QUERY:\n{state['user_query']}\n\n"
        f"REJECTED DRAFT:\n{state.get('draft') or ''}\n\n"
        f"CRITIC SCORES (JSON):\n{scores}\n\n"
        f"ACTIONABLE FEEDBACK:\n{feedback_lines or '(none listed)'}\n\n"
        "Produce the improved draft."
    )

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    draft = response.content if isinstance(response.content, str) else str(response.content)

    record: IterationRecord = {
        "iteration": next_iteration,
        "agent": "refiner",
        "draft": draft,
        "feedback": list(state.get("feedback") or []),
        "pass_fail": False,
        "summary": "Draft rewritten from critic feedback.",
    }
    return {
        "draft": draft,
        "iteration_count": next_iteration,
        "status": "refining",
        "active_agent": "Refiner Agent",
        "audit_trail": _audit_append(state, record),
        "pass_fail": False,
    }


def approve_node(state: GovernanceState) -> dict:
    """Terminal success: draft is cleared to the user."""
    return {
        "status": "approved",
        "active_agent": "Governance Gate",
        "failure_reason": "",
    }


def circuit_breaker_node(state: GovernanceState) -> dict:
    """Terminal failure after max refine loops — liability-preserving fail-closed.

    We do not silently ship the last draft. The UI shows this as
    'Governance Review Failed' plus the last critic payload for audit.
    """
    max_iters = int(state.get("max_iterations") or get_settings().max_governance_iterations)
    reason = (
        "Governance Review Failed: the draft could not be aligned with the "
        f"Enterprise AI Constitution within {max_iters} iteration(s). "
        "The last critic scores and feedback are retained for human review. "
        "Do not treat the last draft as an approved enterprise response."
    )
    return {
        "status": "governance_review_failed",
        "active_agent": "Circuit Breaker",
        "pass_fail": False,
        "failure_reason": reason,
    }
