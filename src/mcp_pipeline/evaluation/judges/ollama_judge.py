from __future__ import annotations

import json
import re
import time

import requests

from mcp_pipeline.evaluation.judges.base import JudgeError, JudgeEvaluation, RubricScores
from mcp_pipeline.evaluation.judges.openai_compatible_judge import _JSON_OBJECT_SHAPE_HINT
from mcp_pipeline.evaluation.prompts import RUBRIC_SYSTEM_PROMPT, build_user_message


class OllamaJudge:
    """Judge para o wrapper FastAPI local que encaminha chamadas ao Ollama."""

    def __init__(
        self,
        judge_id: str,
        model_id: str,
        base_url: str = "http://localhost:8000",
        timeout_seconds: int | None = None,
    ):
        self.judge_id = judge_id
        self.provider = "ollama"
        self.model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def evaluate(self, payload: dict) -> JudgeEvaluation:
        started = time.monotonic()
        try:
            request_body = {
                    "model": self.model_id,
                    "system": RUBRIC_SYSTEM_PROMPT + _JSON_OBJECT_SHAPE_HINT,
                    "prompt": build_user_message(payload),
                    "format": RubricScores.model_json_schema(),
                    "max_tokens": 2048,
                }
            response = requests.post(
                f"{self._base_url}/generate",
                json=request_body,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise JudgeError(f"conexão recusada ao {self.judge_id} ({self._base_url}): {exc}") from exc
        except requests.Timeout as exc:
            raise JudgeError(f"timeout ao chamar {self.judge_id}: {exc}") from exc
        except requests.RequestException as exc:
            detail = f" -- corpo da resposta: {exc.response.text[:1000]}" if exc.response is not None else ""
            raise JudgeError(f"erro HTTP ao chamar {self.judge_id}: {exc}{detail}") from exc

        try:
            result = response.json()
            message_content = result["response"]
        except (ValueError, KeyError, TypeError) as exc:
            raise JudgeError(f"resposta do Ollama não contém JSON válido: {exc}") from exc

        try:
            parsed_json = json.loads(message_content)
        except (TypeError, json.JSONDecodeError):
            json_match = re.search(r"\{.*\}", str(message_content), re.DOTALL)
            if not json_match:
                raise JudgeError("resposta do Ollama não contém JSON de scores válido") from None
            try:
                parsed_json = json.loads(json_match.group(0))
            except json.JSONDecodeError as exc:
                raise JudgeError(f"JSON de scores inválido retornado pelo Ollama: {exc}") from exc

        if isinstance(parsed_json, dict) and "error" in parsed_json:
            raise JudgeError(f"Ollama não conseguiu gerar os scores: {parsed_json['error']}")

        try:
            scores = RubricScores.model_validate(parsed_json)
        except ValueError as exc:
            raise JudgeError(f"resposta de {self.judge_id} não valida contra RubricScores: {exc}") from exc

        input_tokens = result.get("prompt_tokens") or 0
        output_tokens = result.get("completion_tokens") or 0
        return JudgeEvaluation(
            scores=scores,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw_model_version=result.get("model", self.model_id),
            latency_ms=(time.monotonic() - started) * 1000,
        )