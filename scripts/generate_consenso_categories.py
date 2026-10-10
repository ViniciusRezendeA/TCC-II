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
import time
from pathlib import Path

import requests
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.evaluation.judges.base import (
    JudgeBalanceExhausted,
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
#
# NA PRÁTICA, essa chamada única nunca coube no tier gratuito: o payload das ~2685
# justificativas (~392k tokens) excede o teto de 250k tokens/minuto do free tier do Gemini --
# e esse teto é da CONTA, não do modelo (confirmado ao vivo: gemini-3.5-flash-lite bateu no
# mesmo "GenerateContentInputTokensPerModelPerMinute-FreeTier, limit: 250000" que o
# gemini-3.6-flash). Por isso o modelo efetivamente usado nesta etapa é o DeepSeek
# (deepseek-flash, já integrado como juiz pago em judges.yaml): sem teto de tokens/minuto
# publicado, só um teto de concorrência por conta (ver deepseek_judge.py).
DEFAULT_MODEL = "deepseek-flash"

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# Uma chamada com ~392k tokens de entrada demora bem mais que as avaliações individuais de
# ~1-5k tokens da Etapa 3 (timeout de 90s em deepseek_judge.py) -- 10 minutos de margem.
_DEEPSEEK_TIMEOUT_SECONDS = 600

# O payload de entrada tem ~400k tokens (2685 justificativas) -- muito maior que uma avaliação
# individual da Etapa 3 (~1-5k tokens). O _REQUEST_TIMEOUT_MS de gemini_judge.py (60s) é
# calibrado pra essas chamadas pequenas; esta chamada única e grande precisa de bem mais margem.
_REQUEST_TIMEOUT_MS = 300_000  # 5 minutos

# JudgeRateLimited por minuto é teto DE CONTA, agregado entre as chaves -- confirmado ao vivo
# (ver run_sequential_step3.py): todas as 20 chaves levaram o mesmo 429
# (GenerateContentInputTokensPerModelPerMinute-FreeTier) em sequência rápida, porque trocar de
# chave não escapa de um teto por conta. Sem esperar antes de tentar a próxima, o laço de
# fallback simplesmente queima todas as chaves instantaneamente, sem dar tempo da janela de 1
# minuto resetar -- por isso, diferente do loop de 1 tentativa por chave original, agora
# espera e dá até 2 passadas completas pelas chaves antes de desistir.
_DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 20.0
_MAX_PASSADAS_RATE_LIMIT = 2

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


# DeepSeek rejeita response_format="json_schema" pra este modelo (mesma descoberta de
# openai_compatible_judge.py: HTTP 400 "This response_format type is unavailable now"), e
# "json_object" exige um objeto JSON no topo, não um array -- por isso o prompt pede um objeto
# {"categories": [...]} em vez do array puro que o schema do Gemini devolve direto.
_DEEPSEEK_JSON_OBJECT_HINT = (
    '\n\nRespond with a single JSON object with exactly one key, "categories", whose value is '
    "the JSON array described above. Return only the JSON object, no other text."
)


def _call_deepseek(payload: list[dict], model: str, api_key: str) -> list[Category]:
    url = f"{_DEEPSEEK_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION + _DEEPSEEK_JSON_OBJECT_HINT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "max_tokens": 16_000,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=_DEEPSEEK_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.Timeout as e:
        raise JudgeError(f"timeout ao chamar deepseek (limite: {_DEEPSEEK_TIMEOUT_SECONDS}s): {e}") from e
    except requests.RequestException as e:
        if e.response is not None and e.response.status_code == 402:
            raise JudgeBalanceExhausted("deepseek: saldo insuficiente (HTTP 402)") from e
        detail = f" -- corpo da resposta: {e.response.text[:2000]}" if e.response is not None else ""
        raise JudgeError(f"erro HTTP ao chamar deepseek: {e}{detail}") from e

    result = response.json()
    if "error" in result:
        error_info = result.get("error", {})
        raise JudgeError(f"erro da API deepseek: {error_info.get('message', error_info)}")

    choices = result.get("choices", [])
    finish_reason = choices[0].get("finish_reason") if choices else None
    message_content = choices[0].get("message", {}).get("content") if choices else None
    if not message_content:
        raise JudgeError(f"resposta de deepseek não contém conteúdo (finish_reason={finish_reason!r})")

    try:
        parsed = json.loads(message_content)
    except json.JSONDecodeError as e:
        raise JudgeError(f"resposta de deepseek não é JSON válido: {e} -- início: {message_content[:500]}") from e

    categories_raw = parsed.get("categories") if isinstance(parsed, dict) else parsed
    if not isinstance(categories_raw, list):
        raise JudgeError(f"resposta de deepseek não tem a chave 'categories' esperada: {str(parsed)[:500]}")
    return [Category.model_validate(c) for c in categories_raw]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identifica categorias (análise temática indutiva, Gemini, Braun & Clarke 2006) a partir de todas as justificativas de consenso, em uma única chamada."
    )
    parser.add_argument("--input", type=Path, default=None, help="JSONL de entrada (default: data/analysis/consenso_justificativas.jsonl).")
    parser.add_argument("--output", type=Path, default=None, help="JSON de saída (default: data/analysis/consenso_categorias.json).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"model_id (default: {DEFAULT_MODEL}; 'deepseek-flash' usa a DeepSeek API, qualquer outro valor usa Gemini).")
    args = parser.parse_args()

    input_path = args.input or (DATA_DIR / "analysis" / "consenso_justificativas.jsonl")
    output_path = args.output or (DATA_DIR / "analysis" / "consenso_categorias.json")

    if not input_path.exists():
        logger.error("%s não encontrado -- rode `uv run python -m scripts.generate_consenso_justifications` primeiro.", input_path)
        sys.exit(1)

    payload = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info("Carregadas %s justificativas de %s -- enviando em uma única chamada ao modelo %s", len(payload), input_path, args.model)

    use_deepseek = args.model.startswith("deepseek")
    categories: list[Category] | None = None

    if use_deepseek:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            logger.error("DEEPSEEK_API_KEY não definida (ou vazia) no ambiente/.env. Veja .env.example.")
            sys.exit(1)
        try:
            categories = _call_deepseek(payload, args.model, api_key)
        except JudgeRefusal as e:
            logger.error("Recusado pelo provedor (categoria=%s).", e.category)
            sys.exit(1)
        except JudgeBalanceExhausted as e:
            logger.error("%s", e)
            sys.exit(1)
        except JudgeError as e:
            logger.error("Chamada à deepseek falhou: %s", e)
            sys.exit(1)
    else:
        keys_raw = os.environ.get("GOOGLE_API_KEYS", "")
        keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
        if not keys:
            logger.error(
                "GOOGLE_API_KEYS não definida (ou vazia) no ambiente/.env -- mesma variável usada "
                "por scripts/run_sequential_step3.py e scripts/generate_consenso_justifications.py. "
                "Veja .env.example."
            )
            sys.exit(1)

        daily_exhausted: set[int] = set()
        for passada in range(1, _MAX_PASSADAS_RATE_LIMIT + 1):
            if len(daily_exhausted) == len(keys):
                break
            for i, key in enumerate(keys):
                if i in daily_exhausted:
                    continue
                try:
                    categories = _call_gemini(payload, args.model, key)
                    break
                except JudgeRefusal as e:
                    logger.error("Recusado pelo provedor (categoria=%s) -- trocar de chave não ajuda aqui, abortando.", e.category)
                    sys.exit(1)
                except JudgeQuotaExhausted as e:
                    daily_exhausted.add(i)
                    logger.warning("Chave %s esgotou a cota diária, tirando do rodízio: %s", i, e)
                    continue
                except JudgeRateLimited as e:
                    # Teto por minuto agregado por CONTA (ver _DEFAULT_RATE_LIMIT_BACKOFF_SECONDS)
                    # -- trocar de chave sem esperar não escapa dele, por isso espera antes de ir
                    # pra próxima em vez de só "continue" imediato.
                    backoff = e.retry_after_seconds or _DEFAULT_RATE_LIMIT_BACKOFF_SECONDS
                    logger.warning("Chave %s rate-limited (teto de conta), aguardando %.1fs antes da próxima (passada %s/%s): %s", i, backoff, passada, _MAX_PASSADAS_RATE_LIMIT, e)
                    time.sleep(backoff)
                    continue
                except JudgeError as e:
                    logger.warning("Chave %s: erro técnico, tentando a próxima: %s", i, e)
                    continue
            if categories is not None:
                break

        if categories is None:
            logger.error("Todas as %s chaves falharam após %s passada(s) -- nenhuma categoria gerada.", len(keys), _MAX_PASSADAS_RATE_LIMIT)
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
