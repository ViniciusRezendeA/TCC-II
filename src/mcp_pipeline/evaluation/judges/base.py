from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

Likert = Literal[1, 2, 3, 4, 5]


class ComponentScore(BaseModel):
    score: Likert
    reasoning: str = Field(
        description="Brief (1-3 sentence) justification citing specific evidence from the "
        "description (and SOURCE_CODE, if present)."
    )


class RubricScores(BaseModel):
    """The 6-component rubric (Hasan et al. 2026 / this TCC), each scored independently on
    a 5-point Likert scale. No overall/aggregate field on purpose: aggregation (mean of the
    6 components per tool, then across judges) is computed at analysis time (Etapa 5), not
    emitted by the judge itself.
    """

    purpose: ComponentScore
    guidelines: ComponentScore
    limitations: ComponentScore
    parameter_explanation: ComponentScore
    length_completeness: ComponentScore
    examples: ComponentScore


@dataclass
class JudgeEvaluation:
    scores: RubricScores
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    raw_model_version: str
    latency_ms: float


class JudgeError(RuntimeError):
    """Unrecoverable failure after SDK-level retries are exhausted (network, auth, parse failure)."""


class JudgeRefusal(JudgeError):
    """The provider's safety system declined to score the tool -- a real, expected outcome
    for this dataset (a meaningful share of the pilot's tools come from a pentesting MCP
    server), not a bug. Subclasses JudgeError so a caller that only catches JudgeError still
    catches this, but the runner catches JudgeRefusal first to record status="refused"
    instead of status="error" (see pipeline/run_step3.py).
    """

    def __init__(self, category: str | None):
        self.category = category
        super().__init__(f"judge refused (category={category})")


class Judge(Protocol):
    judge_id: str  # stable key, e.g. "claude-haiku-4-5" -- used in filenames/checkpoint keys
    provider: str
    model_id: str

    def evaluate(self, payload: dict) -> JudgeEvaluation: ...
