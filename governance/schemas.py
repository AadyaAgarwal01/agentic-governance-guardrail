"""Pydantic contracts for structured critic output (the automated reward model)."""

from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator


class CriticEvaluation(BaseModel):
    """Strict schema bound to the Governance Critic via with_structured_output.

    This is the RLHF-style reward-model head: numeric scores plus natural-language
    deductions the Refiner must address. Structured output prevents the critic
    from drifting into free-form prose that the graph cannot route on.
    """

    factual_accuracy_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="1–10 score for epistemic humility, no fabrication, scoped claims.",
    )
    neutrality_bias_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="1–10 score for fairness, non-loaded language, multi-perspective coverage.",
    )
    safety_compliance_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="1–10 score for Article I safety / legal compliance.",
    )
    actionable_feedback: List[str] = Field(
        default_factory=list,
        description="Concrete deductions and rewrite instructions. Empty only if pass_fail is True.",
    )
    pass_fail: bool = Field(
        ...,
        description="True only when the draft meets the enterprise pass threshold on all axes.",
    )
    constitution_articles_cited: List[str] = Field(
        default_factory=list,
        description="Article identifiers implicated in deductions, e.g. 'II.3', 'III.1'.",
    )
    summary: str = Field(
        default="",
        description="One-paragraph evaluator rationale for the audit trail.",
    )

    @field_validator(
        "factual_accuracy_score",
        "neutrality_bias_score",
        "safety_compliance_score",
    )
    @classmethod
    def scores_are_integers_in_band(cls, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("Scores must be integers from 1 to 10.")
        return value

    @model_validator(mode="after")
    def fail_requires_feedback(self) -> "CriticEvaluation":
        if not self.pass_fail and not self.actionable_feedback:
            self.actionable_feedback = [
                "Draft failed governance but the critic omitted rewrite instructions. "
                "Tighten claims, add uncertainty labels, rebalance perspectives, "
                "and remove any operationally unsafe detail."
            ]
        return self

    def as_public_dict(self) -> dict:
        return self.model_dump()
