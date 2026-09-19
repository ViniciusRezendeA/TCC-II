from __future__ import annotations

import os

from mcp_pipeline.evaluation.judges.base import JudgeError
from mcp_pipeline.evaluation.judges.openai_compatible_judge import OpenAICompatibleJudge

# https://api.deepseek.com/chat/completions (no /v1 segment -- confirmed against the
# official curl example at api-docs.deepseek.com on 2026-09-17); OpenAICompatibleJudge
# appends "/chat/completions" itself, so base_url stops at the bare host.
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekJudge(OpenAICompatibleJudge):
    """Cloud judge via DeepSeek's official API (OpenAI-compatible).

    Unlike every other cloud judge in this project (Gemini's free tier), this is a PAID
    provider with no free tier confirmed for API usage (only an unverified, time-limited
    signup credit) -- see the cost analysis discussed for this TCC before enabling in bulk.
    Kept disabled by default in judges.yaml; enabling it is a deliberate per-run choice via
    `--judges deepseek-flash`, not something that should happen by editing `enabled: true`
    and forgetting about it.

    model_id "deepseek-flash" (DeepSeek-V4.1-Flash), confirmed live against the model list at
    api-docs.deepseek.com on 2026-09-17. Confirm again against
    `GET https://api.deepseek.com/v1/models` before a batch run: DeepSeek has already renamed
    this model id once (from `deepseek-v4-flash`) and lists both as "functional but
    deprecated" side by side.

    response_format="json_object", not the base class's default "json_schema": a live pilot
    call on 2026-09-17 got back HTTP 400 "This response_format type is unavailable now" for
    `deepseek-flash` with json_schema, despite docs suggesting the model supports strict
    schema enforcement. json_object is DeepSeek's own confirmed-working JSON mode (see
    api-docs.deepseek.com/guides/json_mode) -- output shape now relies on
    RUBRIC_SYSTEM_PROMPT's prose (which already names every key) plus
    OpenAICompatibleJudge's regex-extraction fallback and RubricScores validation, not a
    server-enforced schema. Re-check whether json_schema has become available again before
    reverting this -- it would tighten the guarantee back up.

    No RateLimiter: DeepSeek does not publish a requests-per-minute cap (unlike Gemini's
    free tier); it enforces only a per-account concurrency ceiling (2,500 for this model),
    far above anything `--concurrency` would produce here.

    Lazy loading: credenciais só são validadas na primeira chamada a evaluate().
    """

    def __init__(self, judge_id: str = "deepseek-flash", model_id: str = "deepseek-flash"):
        self._initialized = False
        self.judge_id = judge_id
        self.provider = "deepseek"
        self.model_id = model_id
        self._base_url = None
        self._bearer_token = None
        self._max_tokens = 16_000
        # None (sem timeout) travou ao vivo: uma conexão morta deixa requests.post() bloqueado
        # num socket read para sempre -- mesma classe de bug já corrigida em gemini_judge.py
        # (ver GeminiJudge._REQUEST_TIMEOUT_MS). requests.Timeout já cai no branch de erro
        # comum logo abaixo, então isso só limita o tempo de espera, não muda o tratamento.
        self._timeout_seconds = 90
        self._response_format_type = "json_object"

    def _ensure_initialized(self) -> None:
        """Lazy initialization: valida credenciais apenas quando necessário."""
        if self._initialized:
            return

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise JudgeError("Credencial faltando para DeepSeek: DEEPSEEK_API_KEY")

        self._base_url = _DEEPSEEK_BASE_URL
        self._bearer_token = api_key
        self._initialized = True

    def evaluate(self, payload: dict):
        """Override para garantir que credenciais são carregadas antes da chamada."""
        self._ensure_initialized()
        return super().evaluate(payload)
