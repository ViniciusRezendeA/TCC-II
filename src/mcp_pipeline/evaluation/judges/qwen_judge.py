from __future__ import annotations

import os

from mcp_pipeline.evaluation.judges.base import JudgeError
from mcp_pipeline.evaluation.judges.openai_compatible_judge import OpenAICompatibleJudge


class QwenJudge(OpenAICompatibleJudge):
    """Qwen2.5-14B-Instruct rodando em llama-server local via llama.cpp.
    Especializado em avaliar respostas de outros modelos (LLM-as-a-Judge), com aderência
    a instruções suficiente para validar congruência descrição-código, não apenas
    enriquecer a avaliação.

    Lazy loading: credenciais só são validadas na primeira chamada a evaluate().
    """

    def __init__(self, judge_id: str = "qwen2.5-14b-instruct", model_id: str = "qwen2.5-14b-instruct"):
        self._initialized = False
        self.judge_id = judge_id
        self.provider = "qwen"
        self.model_id = model_id
        self._base_url = None
        self._bearer_token = None
        self._max_tokens = 16_000
        self._timeout_seconds = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization: valida credenciais apenas quando necessário."""
        if self._initialized:
            return

        base_url = os.getenv("QWEN_LLM_BASE_URL")
        bearer_token = os.getenv("QWEN_LLM_BEARER_TOKEN")

        if not base_url or not bearer_token:
            missing = []
            if not base_url:
                missing.append("QWEN_LLM_BASE_URL")
            if not bearer_token:
                missing.append("QWEN_LLM_BEARER_TOKEN")
            raise JudgeError(f"Credenciais faltando para Qwen: {', '.join(missing)}")

        self._base_url = base_url.rstrip("/")
        self._bearer_token = bearer_token
        self._initialized = True

    def evaluate(self, payload: dict):
        """Override para garantir que credenciais são carregadas antes da chamada."""
        self._ensure_initialized()
        # Delegado para a classe base
        return super().evaluate(payload)
