from __future__ import annotations

import threading
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from mcp_pipeline.evaluation.judges.base import (
    JudgeError,
    JudgeEvaluation,
    JudgeQuotaExhausted,
    JudgeRefusal,
    RubricScores,
)
from mcp_pipeline.evaluation.prompts import RUBRIC_SYSTEM_PROMPT, build_user_message

# finish_reason values that mean "the safety/content-policy system intervened", as opposed
# to a normal stop (STOP) or a length cutoff (MAX_TOKENS) -- verified against
# google.genai.types.FinishReason.
_REFUSAL_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


def _is_daily_quota_exhausted(e: genai_errors.APIError) -> bool:
    """True only for a 429 caused by the account's DAILY request quota, never a per-minute
    one -- distinguished via the structured QuotaFailure Google returns (`e.details`, the
    raw response JSON APIError already parses), not by regex-matching `e.message`, which
    isn't a documented/stable format. Confirmed live: the daily-cap violation's `quotaId`
    is "GenerateRequestsPerDayPerProjectPerModel-FreeTier" (contains "PerDay"); the
    per-minute one _RateLimiter is meant to prevent is a distinct "PerMinute" quotaId.
    """
    if e.code != 429:
        return False
    error_details = ((e.details or {}).get("error") or {}).get("details") or []
    for detail in error_details:
        if not str(detail.get("@type", "")).endswith("QuotaFailure"):
            continue
        for violation in detail.get("violations", []):
            if "PerDay" in violation.get("quotaId", ""):
                return True
    return False

# google-genai does NOT retry 429/5xx by default -- verified directly from
# google.genai._api_client.retry_args(): "If None, the 'never retry' stop strategy will be
# used." Unlike the anthropic/openai SDKs (both default to max_retries=2), retries here are
# opt-in via HttpOptions.retry_options, configured natively instead of via a tenacity wrapper.
_DEFAULT_RETRY_OPTIONS = genai_types.HttpRetryOptions(attempts=3)


class _RateLimiter:
    """Paces calls to at most `requests_per_minute`, spaced evenly (60/N seconds apart)
    rather than allowed to burst up to the limit -- run_step3.py calls judge.evaluate()
    from several ThreadPoolExecutor workers at once (--concurrency), and that parallelism
    is otherwise the only throttle in the whole pipeline: nothing paces actual request
    *rate*, only how many are in flight simultaneously. A burst of `concurrency` requests
    fired the instant workers free up blew through Gemini's free-tier RPM (429s observed
    even with --concurrency 3, well under the nominal per-account limit) because fast
    responses meant several bursts happened within one 60s window. Even pacing avoids that
    regardless of --concurrency, at the cost of evaluate() blocking the calling thread.
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


class GeminiJudge:
    """Google Gemini judge via `models.generate_content()` structured output -- verified
    directly against the installed google-genai SDK source (`response.parsed`, config's
    `response_mime_type`/`response_schema`/`system_instruction`, `usage_metadata` fields,
    `FinishReason` enum, and the retry-options behavior noted above).

    ⚠️  FREE TIER LIMITATIONS (IMPORTANTE):
    - RPM varia por modelo e por conta -- não é fixo em 15 (ver requests_per_minute em
      config/judges.yaml, ajustado aos valores reais observados em aistudio.google.com).
    - Quota diária (RPD) também varia por modelo; costuma ser a restrição real para o
      dataset completo, não o RPM (ex.: 500 RPD ÷ 2 cenários = 250 tools/dia).
    - Dataset completo (12.171 tools × 2 cenários = 24.342 chamadas): mesmo no modelo com
      RPD mais alto observado, dezenas de dias -- use Gemini para validação de subset, não
      como juiz do dataset completo.

    Modelo recomendado: gemini-3.6-flash ou gemini-3.5-flash-lite (ver config/judges.yaml).
    NÃO usar variantes "-live-preview": só suportam bidiGenerateContent via WebSocket, não
    o generate_content() usado aqui -- retornam 400 INVALID_ARGUMENT nesse método.
    Recomendação: Use como SECUNDÁRIO para validação (10-20 tools), não para dataset completo.
    Veja scripts/check_gemini_free_tier.py para análise de capacidade diária.
    """

    def __init__(
        self,
        judge_id: str,
        model_id: str,
        max_output_tokens: int = 16_000,
        requests_per_minute: int = 5,
    ):
        self.judge_id = judge_id
        self.provider = "google"
        self.model_id = model_id
        self._max_output_tokens = max_output_tokens
        # Conservative default (5) matches the lowest RPM observed live across free-tier
        # Gemini models (see config/judges.yaml) -- override per judge_id there, since the
        # real per-account limit varies by model and isn't queryable from the API itself.
        self._rate_limiter = _RateLimiter(requests_per_minute)
        # reads GOOGLE_API_KEY (falling back to GEMINI_API_KEY) from env
        self._client = genai.Client(http_options=genai_types.HttpOptions(retry_options=_DEFAULT_RETRY_OPTIONS))

    def evaluate(self, payload: dict) -> JudgeEvaluation:
        self._rate_limiter.wait()
        started = time.monotonic()
        try:
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=build_user_message(payload),
                config=genai_types.GenerateContentConfig(
                    system_instruction=RUBRIC_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=RubricScores,
                    max_output_tokens=self._max_output_tokens,
                ),
            )
        except genai_errors.APIError as e:
            if _is_daily_quota_exhausted(e):
                raise JudgeQuotaExhausted(f"gemini daily quota exhausted: {e.message}") from e
            raise JudgeError(f"gemini API error {e.code}: {e.message}") from e

        finish_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            finish_reason = getattr(finish_reason, "value", finish_reason)  # enum -> str

        scores = response.parsed
        if scores is None:
            if finish_reason in _REFUSAL_FINISH_REASONS:
                raise JudgeRefusal(finish_reason)
            raise JudgeError(f"gemini response had no parsed output (finish_reason={finish_reason!r})")
        if not isinstance(scores, RubricScores):
            scores = RubricScores.model_validate(scores)

        usage = response.usage_metadata
        return JudgeEvaluation(
            scores=scores,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
            cache_read_tokens=(usage.cached_content_token_count or 0) if usage else 0,
            cache_creation_tokens=0,  # Gemini's implicit caching has no separate write-count in usage_metadata
            raw_model_version=response.model_version or self.model_id,
            latency_ms=(time.monotonic() - started) * 1000,
        )
