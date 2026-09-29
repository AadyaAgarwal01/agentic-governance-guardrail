"""LLM factory: Gemini (ChatGoogleGenerativeAI) plus a structured critic head."""

from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from governance.config import Settings, get_settings
from governance.schemas import CriticEvaluation


def build_chat_model(
    settings: Optional[Settings] = None,
    *,
    temperature: Optional[float] = None,
) -> ChatGoogleGenerativeAI:
    """Return a Gemini chat model. Critic calls this with temperature=0.0."""
    cfg = settings or get_settings()
    api_key = cfg.resolved_api_key
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add a Gemini API key, "
            "or export GEMINI_API_KEY / GOOGLE_API_KEY in your shell."
        )
    return ChatGoogleGenerativeAI(
        model=cfg.gemini_model,
        google_api_key=api_key,
        temperature=cfg.gemini_temperature if temperature is None else temperature,
    )


def build_critic_model(settings: Optional[Settings] = None):
    """Critic MUST emit CriticEvaluation — this is the automated reward model.

    with_structured_output binds the Pydantic schema so LangGraph can route on
    pass_fail and scores without regex-parsing free text.
    """
    cfg = settings or get_settings()
    # Near-deterministic judging: low temperature reduces score jitter across runs.
    llm = build_chat_model(cfg, temperature=0.0)
    return llm.with_structured_output(CriticEvaluation)
