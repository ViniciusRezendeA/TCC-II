#!/usr/bin/env python
"""Classifica, via Gemini, cada tool de data/analysis/consenso_justificativas.jsonl contra o
conjunto FIXO de categorias já identificado por scripts/generate_consenso_categories.py
(data/analysis/consenso_categorias.json) -- uma chamada por tool, não em lote: evita o efeito
"lost in the middle" (Liu, N. F., et al. (2023/2024). Lost in the middle: How language models
use long contexts. Transactions of the Association for Computational Linguistics, 12,
157-173.) de uma lista/saída com milhares de itens numa única chamada -- já confirmado na
prática (generate_consenso_categories.py truncou em finish_reason=MAX_TOKENS numa saída de só
8 categorias, bem menor que 2685 classificações).

Base científica (citar na metodologia do TCC, como a segunda fase do desenho indutivo ->
dedutivo iniciado em generate_consenso_categories.py):
    Hsieh, H.-F., & Shannon, S. E. (2005). Three approaches to qualitative content analysis.
    Qualitative Health Research, 15(9), 1277-1288. https://doi.org/10.1177/1049732305276687
"Directed content analysis": parte de um conjunto de categorias já estabelecido (aqui, o
resultado da análise temática indutiva de Braun & Clarke, 2006, gerada no passo anterior) e
codifica todos os dados contra ele, em vez de deixar categorias emergirem de novo.

Classificação multi-label (uma justificativa pode casar mais de uma categoria) -- mesmo
espírito de classificar_motivos()/MOTIVO_KEYWORDS em scripts/analysis_evaluation_report.py
("uma divergência pode casar mais de uma categoria -- é uma junção, não uma classificação
exclusiva"). NONE_CATEGORY ("none") é o fallback explícito quando nenhuma categoria bate --
nunca inventar uma categoria fora da lista fixa, nunca deixar a lista de saída vazia. Qualquer
categoria devolvida fora do conjunto fixo é descartada e avisada (run_with_key_rotation), não
aceita silenciosamente -- ver docstring de ToolCategorization.

Reaproveita a mesma arquitetura de scripts/generate_consenso_justifications.py: rodízio de
múltiplas chaves (GOOGLE_API_KEYS, mesma variável dos juízes reais da Etapa 3 -- este script
roda em momentos diferentes, nunca junto, então não compete por cota na prática), resumível por
tool_uid (pula o que já está no JSONL de saída), um erro técnico não aborta o lote.

Uso:
  uv run python -m scripts.generate_consenso_categorization
  uv run python -m scripts.generate_consenso_categorization --limit 20   # teste pequeno antes do lote completo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    RateLimiter,
)
from mcp_pipeline.evaluation.judges.gemini_judge import (
    _DEFAULT_RETRY_OPTIONS,
    _REQUEST_TIMEOUT_MS,
    _is_daily_quota_exhausted,
    _quota_diagnostics,
    _retry_after_seconds,
)
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("generate_consenso_categorization")

DEFAULT_MODEL = "gemini-3.5-flash-lite"
# Mesmo valor de gemini-3.5-flash-lite em config/judges.yaml -- já validado contra o RPM real
# do free tier para este modelo (mesmo raciocínio de generate_consenso_justifications.py: um
# lote de milhares de chamadas pequenas, não uma chamada única grande como
# generate_consenso_categories.py).
DEFAULT_REQUESTS_PER_MINUTE = 12

# Mesmos valores e mesmo raciocínio de scripts/run_sequential_step3.py/generate_consenso_
# justifications.py: Gemini nem sempre manda RetryInfo.retryDelay num 429 por minuto, e sem um
# teto de tentativas o rodízio giraria para sempre se a conta inteira estivesse throttled por
# mais tempo que alguns backoffs curtos resolvem.
_DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 5.0
_MAX_RATE_LIMIT_RETRIES = 5

# Mesmo conjunto de evaluation/judges/gemini_judge.py -- finish_reason que indica intervenção
# do sistema de segurança/política de conteúdo, não uma parada normal.
_REFUSAL_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}

# Fallback explícito quando nenhuma categoria da lista fixa bate -- nunca uma lista vazia,
# nunca uma categoria inventada fora do conjunto fechado (ver docstring do módulo).
NONE_CATEGORY = "none"


class ToolCategorization(BaseModel):
    tool_uid: str = Field(description="Copied verbatim from the input JSON's tool_uid field.")
    categories: list[str] = Field(
        description=(
            'Every category name (from the fixed list given in the system instruction) whose '
            'description genuinely matches this justification. Use ["none"] as the sole entry '
            "when none of the given categories fits -- never an empty list, never a name "
            "outside the given list."
        )
    )


SYSTEM_INSTRUCTION_TEMPLATE = """You are a qualitative research assistant performing directed content analysis for an undergraduate thesis ("Model Context Protocol (MCP): Avaliação da Qualidade de Descrições de Tools com Base em Contexto de Código-Fonte") that studies whether giving an LLM-as-judge access to a tool's source code changes its evaluation of the tool's natural-language description.

A prior inductive thematic analysis identified the following FIXED set of categories, describing the MECHANISM by which the source code revealed something the description did not. Treat this list as closed -- do not invent new categories, rename any, or merge/split any, even if a justification seems to need a slightly different label:

{categories}

You will receive one JSON object: {{"tool_uid": ..., "justification": ...}}. Assign every category above whose description genuinely matches what THIS justification says -- it may match more than one category (do not force a single label), and should match zero categories only if none of them genuinely fits (in that case, output the literal string "{none_category}" as the sole entry, never an empty list).

Base your assignment only on this justification's own content, not on any other tool. Output ONLY one JSON object, no text before or after it, in exactly this shape:
{{"tool_uid": "<copied verbatim from the input>", "categories": ["<matching category name(s), or \\"{none_category}\\">"]}}"""


def _render_categories(categories: list[dict]) -> str:
    return "\n".join(f"- {c['category']}: {c['description']}" for c in categories)


def build_system_instruction(categories: list[dict]) -> str:
    return SYSTEM_INSTRUCTION_TEMPLATE.format(categories=_render_categories(categories), none_category=NONE_CATEGORY)


class CategorizationClient:
    """Um client Gemini por chave de API -- mesmo papel de JustificationClient em
    generate_consenso_justifications.py: api_key explícito (não a env var global), pra manter N
    instâncias vivas ao mesmo tempo, uma por chave em GOOGLE_API_KEYS. `system_instruction` já
    vem pronto (categorias fixas embutidas uma única vez, não recalculadas por chamada).
    """

    def __init__(self, model_id: str, api_key: str, requests_per_minute: int, system_instruction: str):
        self.model_id = model_id
        self._system_instruction = system_instruction
        self._rate_limiter = RateLimiter(requests_per_minute)
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(retry_options=_DEFAULT_RETRY_OPTIONS, timeout=_REQUEST_TIMEOUT_MS),
        )

    def generate(self, item: dict) -> ToolCategorization:
        self._rate_limiter.wait()
        try:
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=json.dumps(item, ensure_ascii=False),
                config=genai_types.GenerateContentConfig(
                    system_instruction=self._system_instruction,
                    response_mime_type="application/json",
                    response_schema=ToolCategorization,
                    max_output_tokens=1024,
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
        if not isinstance(result, ToolCategorization):
            result = ToolCategorization.model_validate(result)
        return result


def run_with_key_rotation(
    clients: list[CategorizationClient], pending: list[dict], valid_categories: set[str], out_path: Path
) -> None:
    """Mesmo laço de generate_consenso_justifications.py::run_with_key_rotation() -- ver lá os
    comentários completos sobre cada ramo (rotina idêntica: single-flight, troca de chave no
    429/min, tira a chave do rodízio na cota diária, desiste da tool depois de
    _MAX_RATE_LIMIT_RETRIES). Adaptado aqui só pra validar que toda categoria devolvida
    pertence ao conjunto fixo (ou é NONE_CATEGORY) antes de gravar -- qualquer categoria
    inventada fora da lista é descartada e avisada, nunca aceita silenciosamente (ver docstring
    do módulo e de ToolCategorization).
    """
    n = len(clients)
    current = 0
    daily_exhausted: set[int] = set()
    recorded = 0
    rate_limited_hits = 0
    generic_skipped = 0
    n_categoria_invalida = 0

    with open(out_path, "a", encoding="utf-8") as out:
        for i, item in enumerate(pending):
            if len(daily_exhausted) == n:
                logger.error("Todas as %s chaves esgotaram a cota diária -- parando (retome depois do reset).", n)
                break

            result: ToolCategorization | None = None
            rate_limit_attempts = 0
            for _ in range(n + _MAX_RATE_LIMIT_RETRIES):
                if len(daily_exhausted) == n:
                    break
                if current in daily_exhausted:
                    current = (current + 1) % n
                    continue

                client = clients[current]
                try:
                    result = client.generate(item)
                    break
                except JudgeRefusal as e:
                    logger.warning("Tool %s recusada pelo provedor (categoria=%s), pulando.", item["tool_uid"], e.category)
                    break
                except JudgeQuotaExhausted as e:
                    daily_exhausted.add(current)
                    logger.error("Chave %s esgotou a cota diária, tirando do rodízio: %s", current, e)
                    current = (current + 1) % n
                    continue
                except JudgeRateLimited as e:
                    rate_limited_hits += 1
                    rate_limit_attempts += 1
                    if rate_limit_attempts > _MAX_RATE_LIMIT_RETRIES:
                        logger.error("%s 429s de conta seguidos -- desistindo desta tool por agora (fica pendente pra próxima rodada).", rate_limit_attempts)
                        break
                    backoff = e.retry_after_seconds or _DEFAULT_RATE_LIMIT_BACKOFF_SECONDS
                    logger.info("Rate-limited (teto de conta, não da chave %s), aguardando %.1fs antes de tentar de novo", current, backoff)
                    time.sleep(backoff)
                    current = (current + 1) % n
                    continue
                except Exception as e:  # noqa: BLE001 -- não é problema de chave, não adianta rodiziar
                    generic_skipped += 1
                    logger.error("Tool %s: erro técnico: %s", item["tool_uid"], e)
                    break

            if result is not None:
                categorias_validas = [c for c in result.categories if c == NONE_CATEGORY or c in valid_categories]
                if len(categorias_validas) != len(result.categories):
                    n_categoria_invalida += 1
                    logger.warning(
                        "Tool %s: categoria(s) fora da lista fixa descartada(s): %s",
                        result.tool_uid, sorted(set(result.categories) - set(categorias_validas)),
                    )
                if not categorias_validas:
                    categorias_validas = [NONE_CATEGORY]
                out.write(json.dumps({"tool_uid": result.tool_uid, "categories": categorias_validas}, ensure_ascii=False) + "\n")
                out.flush()
                recorded += 1
            # não resolvido (rodízio completo sem sucesso, recusa, ou erro técnico): nada é
            # gravado, a tool fica pendente pra próxima rodada normalmente.

            if (i + 1) % 20 == 0 or i + 1 == len(pending):
                logger.info("%s/%s", i + 1, len(pending))

    logger.info(
        "Concluído: %s de %s tools processadas, %s troca(s) por rate-limit, %s erro(s) técnico(s), "
        "%s categoria(s) inválida(s) descartada(s), %s chave(s) esgotada(s) hoje",
        recorded, len(pending), rate_limited_hits, generic_skipped, n_categoria_invalida, len(daily_exhausted),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classifica cada tool contra o conjunto fixo de categorias (análise de conteúdo dirigida, Hsieh & Shannon 2005, via Gemini)."
    )
    parser.add_argument("--input", type=Path, default=None, help="JSONL de entrada (default: data/analysis/consenso_justificativas.jsonl).")
    parser.add_argument("--categories", type=Path, default=None, help="JSON com a lista fixa de categorias (default: data/analysis/consenso_categorias.json).")
    parser.add_argument("--output", type=Path, default=None, help="JSONL de saída (default: data/analysis/consenso_categorizacao.jsonl).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"model_id do Gemini (default: {DEFAULT_MODEL}).")
    parser.add_argument(
        "--requests-per-minute", type=int, default=DEFAULT_REQUESTS_PER_MINUTE,
        help=f"RPM por chave (default: {DEFAULT_REQUESTS_PER_MINUTE}).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Processa só as N primeiras tools pendentes (teste antes do lote completo).")
    args = parser.parse_args()

    input_path = args.input or (DATA_DIR / "analysis" / "consenso_justificativas.jsonl")
    categories_path = args.categories or (DATA_DIR / "analysis" / "consenso_categorias.json")
    output_path = args.output or (DATA_DIR / "analysis" / "consenso_categorizacao.jsonl")

    if not input_path.exists():
        logger.error("%s não encontrado -- rode `uv run python -m scripts.generate_consenso_justifications` primeiro.", input_path)
        sys.exit(1)
    if not categories_path.exists():
        logger.error("%s não encontrado -- rode `uv run python -m scripts.generate_consenso_categories` primeiro.", categories_path)
        sys.exit(1)

    keys_raw = os.environ.get("GOOGLE_API_KEYS", "")
    keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        logger.error(
            "GOOGLE_API_KEYS não definida (ou vazia) no ambiente/.env -- mesma variável usada "
            "pelos juízes reais da Etapa 3 e pelos demais scripts generate_consenso_*. Veja .env.example."
        )
        sys.exit(1)

    categories = json.loads(categories_path.read_text(encoding="utf-8"))
    valid_categories = {c["category"] for c in categories}
    logger.info("%s categorias fixas carregadas de %s", len(categories), categories_path)

    items = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info("Carregadas %s justificativas de %s", len(items), input_path)

    ja_processadas: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ja_processadas.add(json.loads(line)["tool_uid"])
        logger.info("%s tool(s) já processada(s) em %s, pulando", len(ja_processadas), output_path)

    pending = [item for item in items if item["tool_uid"] not in ja_processadas]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        logger.info("Nada a fazer (tudo já processado).")
        return

    system_instruction = build_system_instruction(categories)
    clients = [CategorizationClient(args.model, key, args.requests_per_minute, system_instruction) for key in keys]
    logger.info("%s tool(s) pendente(s), %s chave(s), modelo %s", len(pending), len(clients), args.model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_with_key_rotation(clients, pending, valid_categories, output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Classificação de categorias falhou")
        sys.exit(1)
