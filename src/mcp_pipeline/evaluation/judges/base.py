from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Annotated, Protocol

from pydantic import BaseModel, Field

# int + ge/le (-> JSON schema "minimum"/"maximum") instead of Literal[1, 2, 3, 4, 5]
# (-> "enum": [1, 2, 3, 4, 5]): Gemini's response_schema conversion only accepts STRING
# enums (google.genai._transformers.process_schema has no branch converting an INTEGER
# enum list), so an int Literal here made GeminiJudge.evaluate() raise a pydantic
# ValidationError on the SDK's own schema validation before any request was sent.
# minimum/maximum is still enough to constrain llama.cpp's grammar-sampled local judges
# to a single digit 1-5, and RubricScores.model_validate() still enforces the range on
# whatever a judge returns.
Likert = Annotated[int, Field(ge=1, le=5)]


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


class JudgeQuotaExhausted(JudgeError):
    """The provider's account-level DAILY quota is exhausted (e.g. Gemini free tier's
    GenerateRequestsPerDay cap) -- unlike a per-minute rate-limit blip (already handled by
    GeminiJudge's own _RateLimiter), every subsequent call today will fail identically until
    the quota resets, so burning through the rest of a --limit batch just produces hundreds
    of duplicate 429s. Subclasses JudgeError so a caller that only catches JudgeError still
    catches this, but run_step3.py catches it first to cancel the judge's remaining pending
    work instead of recording it all as individual errors (see pipeline/run_step3.py).
    """


class RateLimiter:
    """Paces calls to at most `requests_per_minute`, spaced evenly (60/N seconds apart)
    rather than allowed to burst up to the limit -- run_step3.py calls judge.evaluate()
    from several ThreadPoolExecutor workers at once (--concurrency), and that parallelism
    is otherwise the only throttle in the whole pipeline: nothing paces actual request
    *rate*, only how many are in flight simultaneously. A burst of `concurrency` requests
    fired the instant workers free up blew through Gemini's free-tier RPM (429s observed
    even with --concurrency 3, well under the nominal per-account limit) because fast
    responses meant several bursts happened within one 60s window -- then the same thing
    happened to GroqJudge at --concurrency 1 with no limiter at all. Even pacing avoids
    that regardless of --concurrency, at the cost of evaluate() blocking the calling
    thread. Shared across judges (originally Gemini-only) since every free-tier cloud
    provider surveyed for this project turned out to have a tight RPM (Gemini 4-15,
    Cerebras 5, Groq 30).
    """

    def __init__(self, requests_per_minute: int):
        self._interval = 60.0 / requests_per_minute
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_allowed)
            self._next_allowed = start + self._interval
        sleep_for = start - now
        if sleep_for > 0:
            time.sleep(sleep_for)


class Judge(Protocol):
    judge_id: str  # stable key, e.g. "qwen2.5-14b-instruct" -- used in filenames/checkpoint keys
    provider: str
    model_id: str

    def evaluate(self, payload: dict) -> JudgeEvaluation: ...
