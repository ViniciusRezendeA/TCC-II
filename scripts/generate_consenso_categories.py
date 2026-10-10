#!/usr/bin/env python
"""Identifica, via Gemini, as categorias recorrentes de "o que o código revelou que a
descrição não dizia", a partir de TODAS as justificativas geradas por
scripts/generate_consenso_justifications.py (data/analysis/consenso_justificativas.jsonl) --
uma ÚNICA chamada, com as ~2685 justificativas inteiras no payload (não uma por tool, como no
script anterior).

Base científica (citar na seção de metodologia do TCC):
    Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology.
    Qualitative Research in Psychology, 3(2), 77-101.
    https://doi.org/10.1191/1478088706qp063oa

Análise temática INDUTIVA (open coding, bottom-up: as categorias emergem dos dados, sem lista
pré-definida) -- complementa, não substitui, a classificação DEDUTIVA já existente por
palavra-chave (MOTIVO_KEYWORDS/classificar_motivos() em scripts/analysis_evaluation_report.py).
O prompt (SYSTEM_INSTRUCTION abaixo) instrui o modelo a seguir as 5 fases de Braun & Clarke
(familiarização, codificação inicial, busca de temas, revisão de temas, definição e nomeação)
numa única passada -- isto é uma simplificação do método original, que pressupõe múltiplos
coders humanos e revisão iterativa entre eles; declarar essa limitação explicitamente ao citar
o método no TCC (aqui há 1 "coder" automatizado, sem triangulação humana).

Separado dos demais scripts de Gemini de propósito (mesmo raciocínio de
generate_narrative_analysis.py/generate_consenso_justifications.py): usa GOOGLE_API_KEYS (mesma
variável dos juízes reais da Etapa 3 -- este script roda em momentos diferentes, nunca junto,
então não compete por cota na prática).

Diferente de generate_consenso_justifications.py, não há rodízio por TAREFA aqui (é uma única
chamada, não uma por tool) -- em vez disso, tenta cada chave em GOOGLE_API_KEYS em sequência até
uma funcionar, parando na primeira que retornar com sucesso.

Uso:
  uv run python -m scripts.generate_consenso_categories
  uv run python -m scripts.generate_consenso_categories --input data/analysis/consenso_justificativas.jsonl --output data/analysis/consenso_categorias.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.evaluation.judges.base import (
    JudgeError,
    JudgeQuotaExhausted,
    JudgeRateLimited,
    JudgeRefusal,
)
from mcp_pipeline.evaluation.judges.gemini_judge import (
    _DEFAULT_RETRY_OPTIONS,
    _is_daily_quota_exhausted,
    _quota_diagnostics,
    _retry_after_seconds,
)
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("generate_consenso_categories")

# gemini-3.6-flash (não gemini-3.5-flash-lite) de propósito: este script faz uma ÚNICA
# chamada (não um lote de milhares como generate_consenso_justifications.py), então o RPM
# nominal bem mais apertado do 3.6-flash (5 vs 15, ver config/judges.yaml) não é um problema
# aqui -- e é o modelo gratuito mais forte que o projeto já mapeou, o que importa mais pra uma
# análise temática de uma passada só do que pra um lote de avaliações repetitivas.
DEFAULT_MODEL = "gemini-3.6-flash"

# O payload de entrada tem ~400k tokens (2685 justificativas) -- muito maior que uma avaliação
# individual da Etapa 3 (~1-5k tokens). O _REQUEST_TIMEOUT_MS de gemini_judge.py (60s) é
# calibrado pra essas chamadas pequenas; esta chamada única e grande precisa de bem mais margem.
_REQUEST_TIMEOUT_MS = 300_000  # 5 minutos

# Mesmo conjunto de evaluation/judges/gemini_judge.py -- finish_reason que indica intervenção
# do sistema de segurança/política de conteúdo, não uma parada normal.
_REFUSAL_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


class Category(BaseModel):
    category: str = Field(description="A concise (2-5 word) Title Case analytic label for this category.")
    description: str = Field(
        description=(
            "2-3 sentences: what the category captures (the specific kind of discrepancy or "
            "disclosure found in the code), plus one paraphrased illustrative pattern -- never "
            "a specific tool_uid."
        )
    )


# Prompt citado acima (Braun & Clarke, 2006) -- ver docstring do módulo para a referência
# completa a usar na metodologia do TCC.
SYSTEM_INSTRUCTION = """You are a qualitative research assistant conducting a thematic analysis for an undergraduate thesis ("Model Context Protocol (MCP): Avaliação da Qualidade de Descrições de Tools com Base em Contexto de Código-Fonte") that studies whether giving an LLM-as-judge access to a tool's source code changes its evaluation of the tool's natural-language description. For every tool where every judge that evaluated it changed at least one rubric score between the "description_only" and "with_source" scenarios, a short English justification was already written, explaining WHAT the source code revealed that the plain-text description did not.

You will receive a JSON array of {"tool_uid": ..., "justification": ...} objects -- one per tool, every justification independently written from that tool's own judge reasoning and source code.

Task: perform an inductive thematic analysis (open coding, following Braun & Clarke's six-phase method: familiarization, initial coding, searching for themes, reviewing themes, defining and naming themes) across ALL justifications to identify the recurring categories of WHAT KIND of thing the source code tends to reveal that the description does not. Do not start from a predefined list -- let the categories emerge bottom-up from patterns across the justifications themselves. Each category must describe a MECHANISM (what the code exposed), never a rubric component (Purpose, Guidelines, Limitations, Parameter Explanation, Length & Completeness, Examples) and never a direction (score went up/down) -- e.g. "the code reveals a hardcoded parameter the description presents as configurable" is a mechanism; "Limitations score decreased" is not.

Requirements for the final set of categories:
- Between 5 and 15 categories. Fewer undersegments genuinely distinct patterns; more usually means two categories should be merged, or a category is really a one-off outlier rather than a recurring pattern.
- Each category must be grounded in multiple justifications across different tools -- never create a category for a single idiosyncratic case.
- Categories must be mutually distinguishable: if two candidate categories would classify the same justification the same way almost every time, merge them into one.
- Name each category concisely (2-5 words, a short analytic label, Title Case).
- Each description must state, in 2-3 sentences: (a) what the category captures, in terms of the specific kind of discrepancy or disclosure found in the code, and (b) one illustrative, paraphrased example pattern drawn from the justifications -- never quote or name a specific tool_uid.

Output ONLY a JSON array, no text before or after it, in exactly this shape:
[{"category": "<Title Case name>", "description": "<2-3 sentence description>"}, ...]

Write everything in English."""


def _call_gemini(payload: list[dict], model: str, api_key: str) -> list[Category]:
    client = genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(retry_options=_DEFAULT_RETRY_OPTIONS, timeout=_REQUEST_TIMEOUT_MS),
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=json.dumps(payload, ensure_ascii=False),
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=list[Category],
                # 4096 truncou numa chamada real (finish_reason=MAX_TOKENS, sem output parseado,
                # desperdiçando uma tentativa/cota) ao gerar 8 categorias com gemini-3.6-flash --
                # descrições mais longas que as do flash-lite. 16000 (mesmo valor-base de
                # GeminiJudge, ver gemini_judge.py) dá bem mais margem pra até 15 categorias.
                max_output_tokens=16_000,
            ),
        )
    except genai_errors.APIError as e:
        if _is_daily_quota_exhausted(e):
            raise JudgeQuotaExhausted(f"gemini daily quota exhausted: {e.message}") from e
        diagnostics = _quota_diagnostics(e)
        suffix = f" [{diagnostics}]" if diagnostics else ""
        if e.code == 429:
            raise JudgeRateLimited(
                f"gemini API error {e.code}: {e.message}{suffix}", retry_after_seconds=_retry_after_seconds(e)
            ) from e
        raise JudgeError(f"gemini API error {e.code}: {e.message}{suffix}") from e

    finish_reason = None
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
        finish_reason = getattr(finish_reason, "value", finish_reason)

    result = response.parsed
    if result is None:
        if finish_reason in _REFUSAL_FINISH_REASONS:
            raise JudgeRefusal(finish_reason)
        raise JudgeError(f"gemini response had no parsed output (finish_reason={finish_reason!r})")
    return [c if isinstance(c, Category) else Category.model_validate(c) for c in result]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identifica categorias (análise temática indutiva, Gemini, Braun & Clarke 2006) a partir de todas as justificativas de consenso, em uma única chamada."
    )
    parser.add_argument("--input", type=Path, default=None, help="JSONL de entrada (default: data/analysis/consenso_justificativas.jsonl).")
    parser.add_argument("--output", type=Path, default=None, help="JSON de saída (default: data/analysis/consenso_categorias.json).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"model_id do Gemini (default: {DEFAULT_MODEL}).")
    args = parser.parse_args()

    input_path = args.input or (DATA_DIR / "analysis" / "consenso_justificativas.jsonl")
    output_path = args.output or (DATA_DIR / "analysis" / "consenso_categorias.json")

    if not input_path.exists():
        logger.error("%s não encontrado -- rode `uv run python -m scripts.generate_consenso_justifications` primeiro.", input_path)
        sys.exit(1)

    keys_raw = os.environ.get("GOOGLE_API_KEYS", "")
    keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        logger.error(
            "GOOGLE_API_KEYS não definida (ou vazia) no ambiente/.env -- mesma variável usada "
            "por scripts/run_sequential_step3.py e scripts/generate_consenso_justifications.py. "
            "Veja .env.example."
        )
        sys.exit(1)

    payload = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info("Carregadas %s justificativas de %s -- enviando em uma única chamada ao modelo %s", len(payload), input_path, args.model)

    categories: list[Category] | None = None
    for i, key in enumerate(keys):
        try:
            categories = _call_gemini(payload, args.model, key)
            break
        except JudgeRefusal as e:
            logger.error("Recusado pelo provedor (categoria=%s) -- trocar de chave não ajuda aqui, abortando.", e.category)
            sys.exit(1)
        except JudgeQuotaExhausted as e:
            logger.warning("Chave %s esgotou a cota diária, tentando a próxima: %s", i, e)
            continue
        except JudgeRateLimited as e:
            logger.warning("Chave %s rate-limited, tentando a próxima: %s", i, e)
            continue
        except JudgeError as e:
            logger.warning("Chave %s: erro técnico, tentando a próxima: %s", i, e)
            continue

    if categories is None:
        logger.error("Todas as %s chaves falharam -- nenhuma categoria gerada.", len(keys))
        sys.exit(1)

    logger.info("%s categorias identificadas:", len(categories))
    for c in categories:
        logger.info("  - %s", c.category)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([c.model_dump() for c in categories], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Categorias salvas em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Geração de categorias falhou")
        sys.exit(1)
