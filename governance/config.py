"""Runtime configuration loaded from environment variables."""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Central knobs for the governance loop and the backing Gemini LLM.

    Environment variables (see .env.example):
      GEMINI_API_KEY (or GOOGLE_API_KEY), GEMINI_MODEL, GEMINI_TEMPERATURE,
      MAX_GOVERNANCE_ITERATIONS, PASS_THRESHOLD
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    gemini_api_key: str = ""
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.2

    # Circuit breaker: max Researcher/Refiner -> Critic cycles.
    max_governance_iterations: int = Field(default=3, ge=1, le=10)

    # Critic pass rule: every dimension must be >= this threshold.
    pass_threshold: int = Field(default=8, ge=1, le=10)

    @model_validator(mode="after")
    def coalesce_google_keys(self) -> "Settings":
        if not self.gemini_api_key and self.google_api_key:
            self.gemini_api_key = self.google_api_key
        return self

    @computed_field
    @property
    def resolved_api_key(self) -> str:
        return (self.gemini_api_key or self.google_api_key or "").strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
