from __future__ import annotations

import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from mcp_pipeline.evaluation.judges.base import (
    JudgeError,
    JudgeEvaluation,
    JudgeRefusal,
    RubricScores,
)
from mcp_pipeline.evaluation.prompts import RUBRIC_SYSTEM_PROMPT, build_user_message

# finish_reason values that mean "the safety/content-policy system intervened", as opposed
# to a normal stop (STOP) or a length cutoff (MAX_TOKENS) -- verified against
# google.genai.types.FinishReason.
_REFUSAL_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}

# google-genai does NOT retry 429/5xx by default -- verified directly from
# google.genai._api_client.retry_args(): "If None, the 'never retry' stop strategy will be
# used." Unlike the anthropic/openai SDKs (both default to max_retries=2), retries here are
# opt-in via HttpOptions.retry_options, configured natively instead of via a tenacity wrapper.
_DEFAULT_RETRY_OPTIONS = genai_types.HttpRetryOptions(attempts=3)


class GeminiJudge:
    """Google Gemini judge via `models.generate_content()` structured output -- verified
    directly against the installed google-genai SDK source (`response.parsed`, config's
    `response_mime_type`/`response_schema`/`system_instruction`, `usage_metadata` fields,
    `FinishReason` enum, and the retry-options behavior noted above).
    """

    def __init__(self, judge_id: str, model_id: str, max_output_tokens: int = 16_000):
        self.judge_id = judge_id
        self.provider = "google"
        self.model_id = model_id
        self._max_output_tokens = max_output_tokens
        # reads GOOGLE_API_KEY (falling back to GEMINI_API_KEY) from env
        self._client = genai.Client(http_options=genai_types.HttpOptions(retry_options=_DEFAULT_RETRY_OPTIONS))

    def evaluate(self, payload: dict) -> JudgeEvaluation:
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
