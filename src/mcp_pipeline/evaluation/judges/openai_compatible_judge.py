from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import requests

from mcp_pipeline.evaluation.judges.base import (
    JudgeError,
    JudgeEvaluation,
    JudgeRefusal,
    RubricScores,
)
from mcp_pipeline.evaluation.prompts import RUBRIC_SYSTEM_PROMPT, build_user_message

# Grammar-constrains the local llama.cpp server's output to this exact shape (all 6
# components present, each with a 1-5 score and reasoning string) instead of relying on
# generic "valid JSON" mode -- small local judges (Prometheus, Llama) otherwise drift from
# the schema even when the prose instructions describe it correctly.
_RUBRIC_JSON_SCHEMA = RubricScores.model_json_schema()


class OpenAICompatibleJudge:
    """Judge para servidores llama.cpp (llama-server) com API compatível com OpenAI.
    Usado para Prometheus e Llama rodando localmente na rede.

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
    ):
        self.judge_id = judge_id
        self.provider = "local"
        self.model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._bearer_token = bearer_token
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds

    def evaluate(self, payload: dict) -> JudgeEvaluation:
        started = time.monotonic()

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._bearer_token}",
        }

        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(payload)},
            ],
            "max_tokens": self._max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "rubric_scores", "schema": _RUBRIC_JSON_SCHEMA},
            },
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
            raise JudgeError(f"erro HTTP ao chamar {self.judge_id}: {e}") from e

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
