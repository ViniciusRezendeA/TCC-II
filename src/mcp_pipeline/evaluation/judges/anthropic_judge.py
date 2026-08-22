from __future__ import annotations

import time

import anthropic

from mcp_pipeline.evaluation.judges.base import (
    JudgeError,
    JudgeEvaluation,
    JudgeRefusal,
    RubricScores,
)
from mcp_pipeline.evaluation.prompts import RUBRIC_SYSTEM_PROMPT, build_user_message

DEFAULT_MAX_TOKENS = 16_000


class AnthropicJudge:
    """Claude judge via `messages.parse()` structured output -- verified directly against
    the installed anthropic SDK source (Messages.parse signature, ParsedMessage.parsed_output,
    Message.stop_reason/stop_details/RefusalStopDetails.category), not written from memory.

    No manual retry loop: the SDK auto-retries 429/5xx with exponential backoff
    (`max_retries`, default 2) before either of the except clauses below would fire.

    Cache note: `cache_control` on the system block is a no-op (not an error) on judges
    whose model has a minimum cacheable-prefix above this prompt's length -- e.g. Claude
    Haiku 4.5's floor is 4096 tokens, and RUBRIC_SYSTEM_PROMPT is well under that. It's left
    in place unconditionally so the same code path benefits automatically if the configured
    model changes to one with a lower floor (Opus/Sonnet: 1024; Opus 5/Fable 5: 512).
    """

    def __init__(self, judge_id: str, model_id: str, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.judge_id = judge_id
        self.provider = "anthropic"
        self.model_id = model_id
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def evaluate(self, payload: dict) -> JudgeEvaluation:
        started = time.monotonic()
        try:
            response = self._client.messages.parse(
                model=self.model_id,
                max_tokens=self._max_tokens,
                system=[
                    {"type": "text", "text": RUBRIC_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
                ],
                messages=[{"role": "user", "content": build_user_message(payload)}],
                output_format=RubricScores,
            )
        except anthropic.APIConnectionError as e:
            raise JudgeError(f"anthropic connection error: {e}") from e
        except anthropic.APIStatusError as e:
            raise JudgeError(f"anthropic API error {e.status_code}: {e.message}") from e

        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise JudgeRefusal(category)

        scores = response.parsed_output
        if scores is None:
            raise JudgeError("anthropic response had no parsed_output despite a non-refusal stop_reason")

        usage = response.usage
        return JudgeEvaluation(
            scores=scores,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_creation_tokens=usage.cache_creation_input_tokens or 0,
            raw_model_version=response.model,
            latency_ms=(time.monotonic() - started) * 1000,
        )
