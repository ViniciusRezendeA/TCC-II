from __future__ import annotations

import time

import openai

from mcp_pipeline.evaluation.judges.base import (
    JudgeError,
    JudgeEvaluation,
    JudgeRefusal,
    RubricScores,
)
from mcp_pipeline.evaluation.prompts import RUBRIC_SYSTEM_PROMPT, build_user_message


class OpenAIJudge:
    """OpenAI judge via the Responses API's `responses.parse()` structured output -- verified
    directly against the installed openai SDK source (Responses.parse signature,
    ParsedResponse.output_parsed, ResponseUsage/InputTokensDetails, ResponseOutputRefusal).

    No manual retry loop: the SDK auto-retries 429/5xx by default (`max_retries=2`, same
    default as the anthropic SDK), confirmed from `openai/_constants.py`.

    Unlike Claude (a single top-level `stop_reason: "refusal"`), a refusal here surfaces as
    a `type: "refusal"` content block nested inside `response.output` -- there's no
    top-level flag, so a refusal has to be found by scanning `output` when `output_parsed`
    comes back empty.

    Caching is automatic on OpenAI's side for prompts at/above their internal threshold
    (no `cache_control`-equivalent parameter to set) -- nothing to configure here.
    """

    def __init__(self, judge_id: str, model_id: str, max_output_tokens: int = 16_000):
        self.judge_id = judge_id
        self.provider = "openai"
        self.model_id = model_id
        self._max_output_tokens = max_output_tokens
        self._client = openai.OpenAI()  # reads OPENAI_API_KEY from env

    def evaluate(self, payload: dict) -> JudgeEvaluation:
        started = time.monotonic()
        try:
            response = self._client.responses.parse(
                model=self.model_id,
                instructions=RUBRIC_SYSTEM_PROMPT,
                input=build_user_message(payload),
                max_output_tokens=self._max_output_tokens,
                text_format=RubricScores,
            )
        except openai.APIConnectionError as e:
            raise JudgeError(f"openai connection error: {e}") from e
        except openai.APIStatusError as e:
            raise JudgeError(f"openai API error {e.status_code}: {e.message}") from e

        scores = response.output_parsed
        if scores is None:
            refusal_text = _find_refusal(response)
            if refusal_text is not None:
                raise JudgeRefusal(refusal_text)
            raise JudgeError(f"openai response had no output_parsed (status={response.status!r})")

        usage = response.usage
        cache_read = usage.input_tokens_details.cached_tokens if usage and usage.input_tokens_details else 0
        return JudgeEvaluation(
            scores=scores,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            cache_read_tokens=cache_read or 0,
            cache_creation_tokens=0,  # OpenAI doesn't expose a separate cache-write count
            raw_model_version=response.model,
            latency_ms=(time.monotonic() - started) * 1000,
        )


def _find_refusal(response) -> str | None:
    for item in response.output or []:
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "refusal":
                return block.refusal
    return None
