#!/usr/bin/env python
"""Gera, via Gemini, uma justificativa em inglês da mudança de nota para cada tool em
data/analysis/consenso_amostra.jsonl (saída de scripts/export_consenso_sample.py -- já filtrada
pelos 3 passos do consenso entre juízes, com a reasoning de cada juiz e o source_code
anexados).

Reaproveita, em vez de reimplementar, as mesmas regras de rodízio de múltiplas chaves de
scripts/run_sequential_step3.py: um processo só, uma chamada de cada vez (nunca duas em voo),
troca para a próxima chave de GOOGLE_API_KEYS assim que a atual leva um 429 por minuto
(aguardando retry_after_seconds antes, ou um backoff padrão curto), e tira a chave do rodízio
pelo resto da execução se o 429 for a cota DIÁRIA (confirmado pelo `quotaId` estruturado, não
por regex -- ver _is_daily_quota_exhausted() de evaluation/judges/gemini_judge.py, importada
daqui, não reimplementada). Mesmo raciocínio documentado lá: qualquer chave tentada durante uma
janela de throttle por minuto leva o mesmo 429 (é teto de CONTA, agregado entre as chaves),
então o rodízio não escapa disso por si só -- só evita ficar preso numa chave específica se a
conta inteira estiver throttled por mais tempo que alguns backoffs curtos resolvem.

Usa as MESMAS chaves dos juízes reais da Etapa 3 (GOOGLE_API_KEYS, já em .env) -- de propósito
desta vez, não uma variável separada: este script roda em momentos diferentes da Etapa 3
(nunca junto), então não compete por cota diária com ela na prática.

Resumível por tool_uid: lê o JSONL de saída já existente (se houver) antes de começar e pula
qualquer tool_uid já presente nele -- um re-run só processa o que falta, sem checkpoint
separado (não há múltiplos cenários/juízes por tarefa aqui, 1 chamada = 1 tool, então o próprio
arquivo de saída já é o checkpoint).

Uso:
  # .env: GOOGLE_API_KEYS=chave1,chave2,...  (mesma variável usada pelos juízes reais)
  uv run python -m scripts.generate_consenso_justifications
  uv run python -m scripts.generate_consenso_justifications --limit 20   # teste pequeno antes do lote completo
  uv run python -m scripts.generate_consenso_justifications --input data/analysis/consenso_amostra.jsonl --output data/analysis/consenso_justificativas.jsonl
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

logger = setup_logging("generate_consenso_justifications")

DEFAULT_MODEL = "gemini-3.5-flash-lite"
# Mesmo valor de gemini-3.5-flash-lite em config/judges.yaml -- já validado contra o RPM real
# do free tier para este modelo.
DEFAULT_REQUESTS_PER_MINUTE = 12

# Mesmos valores e mesmo raciocínio de scripts/run_sequential_step3.py (ver docstring do
# módulo): Gemini nem sempre manda RetryInfo.retryDelay num 429 por minuto, e sem um teto de
# tentativas o rodízio giraria para sempre se a conta inteira estivesse throttled por mais
# tempo que alguns backoffs curtos resolvem.
_DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 5.0
_MAX_RATE_LIMIT_RETRIES = 5

# Mesmo conjunto de evaluation/judges/gemini_judge.py -- finish_reason que indica intervenção
# do sistema de segurança/política de conteúdo, não uma parada normal.
_REFUSAL_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


class ToolJustification(BaseModel):
    tool_uid: str = Field(description="Copied verbatim from the input JSON's tool_uid field.")
    justification: str = Field(
        description=(
            "A 3-5 sentence English justification of what the source code revealed that the "
            "plain-text description did not, synthesized across every judge's reasoning for "
            "this tool -- not a restatement that 'the score changed' or a verbatim copy of a "
            "single judge's reasoning."
        )
    )


SYSTEM_INSTRUCTION = """You are a research assistant helping analyze results from an undergraduate thesis ("Model Context Protocol (MCP): Avaliação da Qualidade de Descrições de Tools com Base em Contexto de Código-Fonte") that studies whether giving an LLM-as-judge access to a tool's source code changes its evaluation of the tool's natural-language description. Several judge models independently scored the SAME description on a 1-5 Likert scale across 6 rubric components (Purpose, Guidelines, Limitations, Parameter Explanation, Length & Completeness, Examples), once without the source code ("description_only") and once with it ("with_source"). The judges were instructed that the source code may only be used to find a concrete inconsistency between the description and the actual implementation, and to adjust the score accordingly when one is found -- never to raise a score just because code is present.

You will receive one JSON object describing a single tool for which at least one component's score changed between the two scenarios, for every judge that evaluated it unanimously. It has:
- "tool_uid": a unique identifier for the tool.
- "tool_name": the tool's name.
- "avaliacoes": a list of per-judge, per-scenario entries. Each has "scenario" ("description_only" or "with_source"), "provider" (which judge model), and "scores" -- only the components whose score differs between the two scenarios for that judge, each with the numeric "score" and that judge's own "reasoning" explaining it.
- "source_code": the actual source snippet(s) the "with_source" judges saw.

Task: read every judge's reasoning for its changed component(s), cross-reference it against "source_code", and write ONE justification (3-5 sentences) explaining WHAT the source code revealed that the plain-text description did not -- e.g. an undocumented parameter, an undisclosed behavior/output, a missing limitation or error case, or a direct contradiction between the description and the implementation. If judges disagree on the reason, or if a judge's reasoning gives no concrete evidence from the code (just a vaguer/stricter reading of the same text), say so plainly instead of inventing a justification.

Ground every claim in specific details from the reasoning and/or the source code. Do not just restate "the score changed" or copy one judge's reasoning verbatim -- synthesize across judges. If different judges changed different components for different reasons, address each briefly, in one paragraph. Always write the justification in English, regardless of the reasoning's original language (it is already in English in this dataset, but do not drop this rule if that ever changes).

Be conservative per component: a strong, well-evidenced finding for one judge/component (e.g. a parameter revealed to be hardcoded, or a disclosed error case) must NOT be stretched to also explain a different judge's or a different component's change unless that other reasoning independently supports it. When a specific score change's reasoning is vague or generic (e.g. "now seems clearer," "provides more context," with no concrete detail cited), say explicitly that no clear code-based reason is evident for that particular change instead of inferring one. Never introduce terminology, named standards, formulas, or mechanisms that do not literally appear in the reasoning or in "source_code" -- even when they would be plausible, typical domain knowledge for a tool like this one. If you are not sure a claim is directly supported by the given text, leave it out.

Output ONLY one JSON object, no text before or after it, in exactly this shape:
{"tool_uid": "<the tool_uid from the input, verbatim>", "justification": "<your justification>"}"""


class JustificationClient:
    """Um client Gemini por chave de API -- mesma necessidade de GeminiJudge de manter N
    instâncias vivas ao mesmo tempo (api_key explícito em vez de ler a env var global), uma
    por chave em GOOGLE_API_KEYS.
    """

    def __init__(self, model_id: str, api_key: str, requests_per_minute: int):
        self.model_id = model_id
        self._rate_limiter = RateLimiter(requests_per_minute)
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(retry_options=_DEFAULT_RETRY_OPTIONS, timeout=_REQUEST_TIMEOUT_MS),
        )

    def generate(self, tool: dict) -> ToolJustification:
        self._rate_limiter.wait()
        try:
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=json.dumps(tool, ensure_ascii=False),
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=ToolJustification,
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
        if not isinstance(result, ToolJustification):
            result = ToolJustification.model_validate(result)
        return result


def run_with_key_rotation(clients: list[JustificationClient], pending: list[dict], out_path: Path) -> None:
    """Mesmo laço de scripts/run_sequential_step3.py::run_with_key_rotation(), adaptado pra
    uma tarefa = uma chamada (sem cenário/juiz, que lá vinham do schema da Etapa 3).
    """
    n = len(clients)
    current = 0
    daily_exhausted: set[int] = set()
    recorded = 0
    rate_limited_hits = 0
    generic_skipped = 0

    with open(out_path, "a", encoding="utf-8") as out:
        for i, tool in enumerate(pending):
            if len(daily_exhausted) == n:
                logger.error("Todas as %s chaves esgotaram a cota diária -- parando (retome depois do reset).", n)
                break

            result: ToolJustification | None = None
            rate_limit_attempts = 0
            for _ in range(n + _MAX_RATE_LIMIT_RETRIES):
                if len(daily_exhausted) == n:
                    break
                if current in daily_exhausted:
                    current = (current + 1) % n
                    continue

                client = clients[current]
                try:
                    result = client.generate(tool)
                    break
                except JudgeRefusal as e:
                    logger.warning("Tool %s recusada pelo provedor (categoria=%s), pulando.", tool["tool_uid"], e.category)
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
                    logger.error("Tool %s: erro técnico: %s", tool["tool_uid"], e)
                    break

            if result is not None:
                out.write(json.dumps({"tool_uid": result.tool_uid, "justification": result.justification}, ensure_ascii=False) + "\n")
                out.flush()
                recorded += 1
            # não resolvido (rodízio completo sem sucesso, recusa, ou erro técnico): nada é
            # gravado, a tool fica pendente pra próxima rodada normalmente.

            if (i + 1) % 20 == 0 or i + 1 == len(pending):
                logger.info("%s/%s", i + 1, len(pending))

    logger.info(
        "Concluído: %s de %s tools processadas, %s troca(s) por rate-limit, %s erro(s) técnico(s), %s chave(s) esgotada(s) hoje",
        recorded, len(pending), rate_limited_hits, generic_skipped, len(daily_exhausted),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera justificativas (Gemini) da mudança de nota pra cada tool com consenso entre juízes.")
    parser.add_argument("--input", type=Path, default=None, help="JSONL de entrada (default: data/analysis/consenso_amostra.jsonl).")
    parser.add_argument("--output", type=Path, default=None, help="JSONL de saída (default: data/analysis/consenso_justificativas.jsonl).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"model_id do Gemini (default: {DEFAULT_MODEL}).")
    parser.add_argument(
        "--requests-per-minute", type=int, default=DEFAULT_REQUESTS_PER_MINUTE,
        help=f"RPM por chave (default: {DEFAULT_REQUESTS_PER_MINUTE}, mesmo valor de gemini-3.5-flash-lite em config/judges.yaml).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Processa só as N primeiras tools pendentes (teste antes do lote completo).")
    args = parser.parse_args()

    input_path = args.input or (DATA_DIR / "analysis" / "consenso_amostra.jsonl")
    output_path = args.output or (DATA_DIR / "analysis" / "consenso_justificativas.jsonl")

    if not input_path.exists():
        logger.error("%s não encontrado -- rode `uv run python -m scripts.export_consenso_sample` primeiro.", input_path)
        sys.exit(1)

    keys_raw = os.environ.get("GOOGLE_API_KEYS", "")
    keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        logger.error(
            "GOOGLE_API_KEYS não definida (ou vazia) no ambiente/.env -- mesma variável usada "
            "por scripts/run_sequential_step3.py pros juízes reais da Etapa 3 (uma ou mais "
            "chaves, separadas por vírgula). Veja .env.example."
        )
        sys.exit(1)

    tools = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info("Carregadas %s tools de %s", len(tools), input_path)

    ja_processadas: set[str] = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ja_processadas.add(json.loads(line)["tool_uid"])
        logger.info("%s tool(s) já processada(s) em %s, pulando", len(ja_processadas), output_path)

    pending = [t for t in tools if t["tool_uid"] not in ja_processadas]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        logger.info("Nada a fazer (tudo já processado).")
        return

    clients = [JustificationClient(args.model, key, args.requests_per_minute) for key in keys]
    logger.info("%s tool(s) pendente(s), %s chave(s), modelo %s", len(pending), len(clients), args.model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_with_key_rotation(clients, pending, output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Geração de justificativas falhou")
        sys.exit(1)
