#!/usr/bin/env python
"""Etapa 3 com rodízio de chaves: um único processo, uma tarefa (tool, cenário) de cada vez
-- nunca duas chamadas em voo ao mesmo tempo --, trocando para a próxima chave em
GOOGLE_API_KEYS assim que a atual leva um 429, em vez de esperar o limitador dela mesma.

Por que não scripts/run_parallel_step3.py (N processos simultâneos): confirmado ao vivo que
mesmo com as chaves pertencendo a projetos Google diferentes, rodar várias ao mesmo tempo
gera 429 (GenerateRequestsPerMinutePerProjectPerModel) numa taxa que sobe com o volume total
somado entre as chaves, não com o volume de cada uma isolada -- e que some por completo
quando só uma chave está ativa por vez (--max-parallel 1 bateu 100% de sucesso). Este script
é a versão que aproveita isso: zero simultaneidade (então nada do throttling agregado
observado), mas sem desperdiçar o tempo parado de uma chave só.

JudgeRateLimited (429 por minuto) é teto DE CONTA, agregado entre as 20 chaves -- confirmado
ao vivo (ver acima) que qualquer chave tentada durante uma janela de throttle leva o mesmo
429, então trocar de chave na hora não escapa dele, só paga de novo a espera do rate limiter
daquela chave e o round-trip HTTP para nada. Visto isso acontecer ao vivo: uma única tarefa
varrendo as 20 chaves em sequência, todas rejeitadas, cada uma já gastando ~5-14s de espera
do próprio limiter antes de confirmar o 429 -- até ~1-2min perdidos numa tarefa só. A correção
é honrar retry_after_seconds (ou um default curto quando o provedor não manda um) antes de
tentar de novo, e desistir depois de algumas tentativas seguidas em vez de rodar para sempre
se a conta inteira estiver throttled por um tempo mais longo. Cota diária confirmada
(JudgeQuotaExhausted) É por chave/projeto -- essa sim tira a chave do rodízio pelo resto da
execução (não adianta tentar de novo hoje).

Reaproveita quase tudo de pipeline/run_step3.py (tool_uid_for, checkpoint_key, should_skip,
_base_record) e o mesmo checkpoint/jsonl de saída de uma rodada normal de uma chave só --
não precisa de checkpoint por índice como run_parallel_step3.py, porque não há mais de um
processo escrevendo ao mesmo tempo.

Uso:
  # .env: GOOGLE_API_KEYS=chave1,chave2,chave3,...
  uv run python scripts/run_sequential_step3.py --judge gemini-3.5-flash-lite
  uv run python scripts/run_sequential_step3.py --judge gemini-3.5-flash-lite --limit 100
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from mcp_pipeline.collection.checkpoint import Checkpoint
from mcp_pipeline.config import DATA_DIR, LOGS_DIR, STATE_DIR, ensure_dirs
from mcp_pipeline.evaluation.judges.base import Judge, JudgeQuotaExhausted, JudgeRateLimited, JudgeRefusal
from mcp_pipeline.evaluation.judges.registry import build_judge, provider_for
from mcp_pipeline.evaluation.payload import build_payload, repo_src_root_for
from mcp_pipeline.extraction.models import CallGraphNode, ToolRecord
from mcp_pipeline.logging_setup import setup_logging
from mcp_pipeline.pipeline.run_step3 import (
    SCENARIOS,
    _base_record,
    checkpoint_key,
    load_dataset_rows,
    should_skip,
    tool_uid_for,
)

logger = setup_logging("run_sequential_step3")

_PROVIDER_API_KEY_ENV = {
    "google": "GOOGLE_API_KEY",
}

# Gemini nem sempre manda RetryInfo.retryDelay num 429 por minuto (visto ao vivo: boa parte
# vem com retry_after_seconds=None) -- este é o backoff usado nesse caso.
_DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 5.0
# Tentativas seguidas de JudgeRateLimited antes de desistir da tarefa por agora (fica pendente
# para a próxima rodada) -- evita rodar para sempre se a conta inteira estiver throttled por
# um tempo mais longo que alguns backoffs curtos resolvem.
_MAX_RATE_LIMIT_RETRIES = 5


def build_pending(rows: list[dict], scenarios: tuple[str, ...], judge: Judge, checkpoint: Checkpoint, retry_failed: bool) -> list[tuple[dict, dict, str]]:
    """Mesma construção de pending que run_step3.py::run_judge faz -- `judge` aqui só serve
    para preencher provider/model_id/judge_id no registro-base, todas as instâncias
    (uma por chave) compartilham o mesmo judge_id, então o checkpoint/jsonl são os mesmos
    independente de qual chave de fato atendeu cada tarefa.
    """
    pending: list[tuple[dict, dict, str]] = []
    for row in rows:
        tool_uid = tool_uid_for(row)
        for scenario in scenarios:
            key = checkpoint_key(tool_uid, scenario, judge.judge_id)
            if should_skip(checkpoint, key, retry_failed):
                continue

            tool = ToolRecord.from_dict(row["tool"])
            call_graph = CallGraphNode.from_dict(row["call_graph"])
            name_with_owner = row["repo"]["name_with_owner"]
            payload = build_payload(
                tool, call_graph, repo_src_root_for(name_with_owner), name_with_owner,
                include_source=(scenario == "with_source"),
            )
            source_code = payload.get("SOURCE_CODE")
            source_code_sha256 = hashlib.sha256(source_code.encode("utf-8")).hexdigest() if source_code else None
            record = _base_record(row, tool_uid, scenario, judge, source_code_sha256)
            pending.append((record, payload, key))
    return pending


def run_with_key_rotation(
    judges: list[Judge],
    pending: list[tuple[dict, dict, str]],
    checkpoint: Checkpoint,
) -> None:
    judge_id = judges[0].judge_id
    out_path = DATA_DIR / "evaluations" / f"{judge_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors_log = LOGS_DIR / f"step3_errors_{judge_id}.jsonl"
    errors_log.parent.mkdir(parents=True, exist_ok=True)

    n = len(judges)
    current = 0
    daily_exhausted: set[int] = set()
    recorded = 0
    rate_limited_hits = 0
    generic_skipped = 0

    with open(out_path, "a", encoding="utf-8") as out:
        for i, (record, payload, key) in enumerate(pending):
            if len(daily_exhausted) == n:
                logger.error(
                    "[%s] todas as %s chaves esgotaram a cota diária -- parando (o que sobrou "
                    "não fica marcado, um re-run depois que a cota resetar já retoma sozinho).",
                    judge_id, n,
                )
                break

            resolved = False
            rate_limit_attempts = 0
            for _ in range(n + _MAX_RATE_LIMIT_RETRIES):
                if len(daily_exhausted) == n:
                    break
                if current in daily_exhausted:
                    current = (current + 1) % n
                    continue

                judge = judges[current]
                try:
                    result = judge.evaluate(payload)
                    record["status"] = "ok"
                    record["scores"] = result.scores.model_dump()
                    record["judge"]["raw_model_version"] = result.raw_model_version
                    record["usage"] = {
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "cache_read_input_tokens": result.cache_read_tokens,
                        "cache_creation_input_tokens": result.cache_creation_tokens,
                    }
                    record["latency_ms"] = result.latency_ms
                    resolved = True
                    break
                except JudgeRefusal as e:
                    record["status"] = "refused"
                    record["error_detail"] = f"category={e.category}"
                    resolved = True
                    break
                except JudgeQuotaExhausted as e:
                    daily_exhausted.add(current)
                    logger.error("[%s] chave %s esgotou a cota diária, tirando do rodízio: %s", judge_id, current, e)
                    current = (current + 1) % n
                    continue
                except JudgeRateLimited as e:
                    rate_limited_hits += 1
                    rate_limit_attempts += 1
                    if rate_limit_attempts > _MAX_RATE_LIMIT_RETRIES:
                        logger.error(
                            "[%s] %s 429s de conta seguidos -- desistindo desta tarefa por agora (fica pendente para a próxima rodada)",
                            judge_id, rate_limit_attempts,
                        )
                        break
                    backoff = e.retry_after_seconds or _DEFAULT_RATE_LIMIT_BACKOFF_SECONDS
                    logger.info(
                        "[%s] rate-limited (teto de conta, não da chave %s), aguardando %.1fs antes de tentar de novo",
                        judge_id, current, backoff,
                    )
                    time.sleep(backoff)
                    current = (current + 1) % n
                    continue
                except Exception as e:  # noqa: BLE001 -- não é problema de chave, não adianta rodiziar
                    generic_skipped += 1
                    with open(errors_log, "a", encoding="utf-8") as ef:
                        ef.write(json.dumps({"tool_uid": record["tool_uid"], "scenario": record["scenario"], "error": str(e)}, ensure_ascii=False) + "\n")
                    break

            if resolved:
                record["evaluated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                checkpoint.set(key, {"status": record["status"]})
                recorded += 1
            # não resolvido (rodízio completo sem sucesso, ou erro técnico): nada é gravado,
            # a tarefa some da lista de pendentes na próxima rodada normalmente.

            if (i + 1) % 20 == 0 or i + 1 == len(pending):
                logger.info("[%s] %s/%s", judge_id, i + 1, len(pending))

    logger.info(
        "[%s] concluído: %s registradas de %s pendentes, %s trocas por rate-limit, %s erro(s) técnico(s), %s chave(s) esgotada(s) hoje",
        judge_id, recorded, len(pending), rate_limited_hits, generic_skipped, len(daily_exhausted),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 3 com rodízio de chaves (single-flight, troca no 429).")
    parser.add_argument("--judge", type=str, required=True, help="Um único judge_id de config/judges.yaml.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIOS))
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()

    scenarios = tuple(args.scenarios.split(","))
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        parser.error(f"cenário(s) desconhecido(s): {sorted(unknown)}. Válidos: {SCENARIOS}")

    provider = provider_for(args.judge)
    env_var = _PROVIDER_API_KEY_ENV.get(provider)
    if env_var is None:
        logger.error("Provedor %r não tem rodízio de chave configurado (só cloud com API key).", provider)
        sys.exit(1)

    keys_env_name = f"{env_var}S"
    keys_raw = os.environ.get(keys_env_name, "")
    keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        logger.error("%s não definido (ou vazio) no ambiente/.env.", keys_env_name)
        sys.exit(1)

    ensure_dirs()
    dataset_path = args.dataset or (DATA_DIR / "dataset.jsonl")
    if not dataset_path.exists():
        logger.error("dataset.jsonl não encontrado em %s.", dataset_path)
        sys.exit(1)

    rows = load_dataset_rows(dataset_path, limit=args.limit)
    logger.info("Etapa 3 (rodízio): %s tools carregadas, %s chave(s) em %s", len(rows), len(keys), keys_env_name)

    judges = [build_judge(args.judge, api_key=key) for key in keys]

    checkpoint = Checkpoint(STATE_DIR / "step3_progress.json")
    pending = build_pending(rows, scenarios, judges[0], checkpoint, args.retry_failed)
    if not pending:
        logger.info("[%s] nada a fazer (tudo já concluído)", args.judge)
        return

    logger.info("[%s] %s avaliações pendentes", args.judge, len(pending))
    run_with_key_rotation(judges, pending, checkpoint)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Etapa 3 (rodízio) falhou")
        sys.exit(1)
