from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import requests

from mcp_pipeline.evaluation.judges.base import (
    JudgeBalanceExhausted,
    JudgeError,
    JudgeEvaluation,
    JudgeRefusal,
    RubricScores,
)
from mcp_pipeline.evaluation.prompts import RUBRIC_SYSTEM_PROMPT, build_user_message

# Grammar-constrains the local llama.cpp server's output to this exact shape (all 6
# components present, each with a 1-5 score and reasoning string) instead of relying on
# generic "valid JSON" mode -- small local judges (Qwen, Llama) otherwise drift from
# the schema even when the prose instructions describe it correctly.
_RUBRIC_JSON_SCHEMA = RubricScores.model_json_schema()

# Appended only for providers using response_format="json_object" (no server-side schema
# enforcement available) -- RUBRIC_SYSTEM_PROMPT itself stays untouched (same rubric text,
# same PROMPT_VERSION, for every judge) since this is a technical output-shape instruction,
# not a change to the rubric being evaluated. Needed because a live DeepSeek pilot
# (2026-09-17) returned bare ints ({"purpose": 5, ...}) under json_object without this --
# the prose telling it to "return a Likert score and a reasoning" doesn't by itself imply
# the score/reasoning pair must be nested under each component key.
_JSON_OBJECT_SHAPE_HINT = (
    "\n\nRespond with a single JSON object with exactly these 6 keys: purpose, guidelines, "
    "limitations, parameter_explanation, length_completeness, examples. Each key's value "
    "must itself be a JSON object with exactly two fields: \"score\" (integer 1-5) and "
    "\"reasoning\" (string). Example shape for one key: "
    '{"purpose": {"score": 4, "reasoning": "..."}}. Return only the JSON object, no other text.'
)


class OpenAICompatibleJudge:
    """Judge para servidores llama.cpp (llama-server) com API compatível com OpenAI.
    Usado para Qwen e Llama rodando localmente na rede.

    Requisições HTTP diretas (não usa SDK OpenAI) para ter controle total sobre headers
    de autenticação Bearer e timeouts de conexão local.
    """

    def __init__(
        self,
        judge_id: str,
        model_id: str,
        base_url: str,
        bearer_token: str,
        max_tokens: int = 16_000,
        timeout_seconds: int | None = None,
        response_format: str = "json_schema",
    ):
        self.judge_id = judge_id
        self.provider = "local"
        self.model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._bearer_token = bearer_token
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        # "json_schema" (default) grammar-constrains local llama.cpp servers to the exact
        # RubricScores shape. Some cloud providers reject that response_format type outright
        # (confirmed live for DeepSeek's deepseek-flash on 2026-09-17: 400 "This
        # response_format type is unavailable now") -- those subclasses pass "json_object"
        # instead, relying on RUBRIC_SYSTEM_PROMPT already naming every key ("purpose",
        # "guidelines", ...) and mentioning "JSON object" to steer the shape, plus the
        # regex-extraction fallback and RubricScores.model_validate() below to catch drift.
        self._response_format_type = response_format

    def evaluate(self, payload: dict) -> JudgeEvaluation:
        started = time.monotonic()

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._bearer_token}",
        }

        response_format_type = getattr(self, "_response_format_type", "json_schema")
        if response_format_type == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "rubric_scores", "schema": _RUBRIC_JSON_SCHEMA},
            }
            system_content = RUBRIC_SYSTEM_PROMPT
        else:
            response_format = {"type": response_format_type}
            system_content = RUBRIC_SYSTEM_PROMPT + _JSON_OBJECT_SHAPE_HINT

        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": build_user_message(payload)},
            ],
            "max_tokens": self._max_tokens,
            "response_format": response_format,
        }

        try:
            response = requests.post(url, headers=headers, json=body, timeout=self._timeout_seconds)
            response.raise_for_status()
        except requests.ConnectionError as e:
            raise JudgeError(f"conexão recusada ao {self.judge_id} ({self._base_url}): {e}") from e
        except requests.Timeout as e:
            timeout_str = f"limite: {self._timeout_seconds}s" if self._timeout_seconds else "sem limite"
            raise JudgeError(f"timeout ao chamar {self.judge_id} ({timeout_str}): {e}") from e
        except requests.RequestException as e:
            # HTTP 402 ("Insufficient Balance") is the standard signal, across OpenAI-
            # compatible prepaid providers, that the account hit $0 -- confirmed for DeepSeek
            # at api-docs.deepseek.com's error-code table. Every subsequent call would fail
            # identically until a human tops up, so this gets its own exception (same
            # cancel-the-rest-of-the-batch treatment as JudgeQuotaExhausted) instead of
            # falling into the generic JudgeError branch below and burning through the whole
            # remaining batch one doomed request at a time.
            if e.response is not None and e.response.status_code == 402:
                raise JudgeBalanceExhausted(f"{self.judge_id}: saldo insuficiente (HTTP 402)") from e
            # str(e) alone is just "400 Client Error: Bad Request for url: ..." -- no hint of
            # *why*. The provider's error body (e.g. "model does not support response_format:
            # json_schema") is what actually lets a human (or the next debugging pass) fix a
            # failing judge instead of guessing from a bare status code.
            detail = f" -- corpo da resposta: {e.response.text[:1000]}" if e.response is not None else ""
            raise JudgeError(f"erro HTTP ao chamar {self.judge_id}: {e}{detail}") from e

        try:
            result = response.json()
        except json.JSONDecodeError as e:
            raise JudgeError(f"resposta de {self.judge_id} não é JSON válido: {e}") from e

        if "error" in result:
            error_info = result.get("error", {})
            error_msg = error_info.get("message", str(error_info))
            raise JudgeError(f"erro da API {self.judge_id}: {error_msg}")

        finish_reason = None
        choices = result.get("choices", [])
        if choices:
            finish_reason = choices[0].get("finish_reason")
            if finish_reason == "content_filter":
                raise JudgeRefusal("content_filter")

        message_content = None
        if choices:
            message = choices[0].get("message", {})
            message_content = message.get("content")

        if not message_content:
            raise JudgeError(f"resposta de {self.judge_id} não contém conteúdo (finish_reason={finish_reason})")

        try:
            parsed_json = json.loads(message_content)
        except json.JSONDecodeError:
            # Try to extract JSON from response (handles cases where model wraps JSON in explanation text)
            json_match = re.search(r'\{.*\}', message_content, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(0))
                except json.JSONDecodeError as e:
                    raise JudgeError(f"resposta de {self.judge_id} não é JSON válido no content: {e}") from e
            else:
                raise JudgeError(f"resposta de {self.judge_id} não contém JSON válido: {message_content[:200]}") from None

        try:
            scores = RubricScores.model_validate(parsed_json)
        except ValueError as e:
            raise JudgeError(f"resposta de {self.judge_id} não valida contra RubricScores: {e}") from e

        usage = result.get("usage", {})
        return JudgeEvaluation(
            scores=scores,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw_model_version=result.get("model", self.model_id),
            latency_ms=(time.monotonic() - started) * 1000,
        )
