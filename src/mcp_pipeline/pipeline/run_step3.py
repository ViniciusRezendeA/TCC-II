from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mcp_pipeline.collection.checkpoint import Checkpoint
from mcp_pipeline.config import DATA_DIR, LOGS_DIR, STATE_DIR, ensure_dirs
from mcp_pipeline.evaluation.judges.base import Judge, JudgeBalanceExhausted, JudgeQuotaExhausted, JudgeRefusal
from mcp_pipeline.evaluation.judges.registry import load_judges
from mcp_pipeline.evaluation.payload import build_payload, repo_src_root_for
from mcp_pipeline.evaluation.prompts import PROMPT_VERSION
from mcp_pipeline.extraction.models import CallGraphNode, ToolRecord
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("step3")

SCENARIOS = ("description_only", "with_source")


def load_dataset_rows(dataset_path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def tool_uid_for(row: dict) -> str:
    """Stable semantic key (repo + qualified_name + source location), not a row index into
    dataset.jsonl -- a row-index key would silently misfire if assemble_dataset.py is ever
    re-run with a different repo set. Also the pairing key Etapa 5's Wilcoxon test needs
    ("same tool, same model, scenario A vs B") and the join key for the human spot-check.
    """
    loc = row["tool"]["source_location"]
    return f"{row['repo']['name_with_owner']}::{row['tool']['qualified_name']}::{loc['file']}:{loc['start_line']}"


def checkpoint_key(tool_uid: str, scenario: str, judge_id: str) -> str:
    """PROMPT_VERSION is part of the key on purpose: bumping it whenever the rubric text or
    RubricScores schema changes materially makes every old key inert (never looked up again)
    without deleting the old results -- they stay in the output JSONL tagged with their own
    prompt_version, side by side with the new ones.
    """
    return f"{tool_uid}::{scenario}::{judge_id}::{PROMPT_VERSION}"


def should_skip(checkpoint: Checkpoint, key: str, retry_failed: bool) -> bool:
    entry = checkpoint.get(key)
    if entry is None:
        return False
    if entry.get("status") in ("ok", "refused"):
        return True
    return not retry_failed  # status == "error"


def shard_for(tool_uid: str, num_keys: int) -> int:
    """Deterministic partition of a tool_uid into [0, num_keys) via a stable hash, so the
    same dataset + num_keys always recomputes the identical split -- no partition file to
    generate upfront or keep in sync with dataset.jsonl. This is what lets
    scripts/run_parallel_step3.py fan a single --judges run out across N API keys: each of
    the N processes filters the same dataset.jsonl down to its own disjoint slice just by
    passing a different --key-index, with nothing shared between them but read-only input.
    """
    digest = hashlib.sha256(tool_uid.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_keys


def _base_record(row: dict, tool_uid: str, scenario: str, judge: Judge, source_code_sha256: str | None) -> dict:
    return {
        "schema_version": "1.0",
        "prompt_version": PROMPT_VERSION,
        "tool_uid": tool_uid,
        "repo": {
            "name_with_owner": row["repo"]["name_with_owner"],
            "stargazer_count": row["repo"].get("stargazer_count"),
            "primary_language": row["repo"].get("primary_language"),
        },
        "tool": {
            "name": row["tool"]["name"],
            "qualified_name": row["tool"]["qualified_name"],
            "description": row["tool"]["description"],
        },
        "scenario": scenario,
        "judge": {
            "id": judge.judge_id,
            "provider": judge.provider,
            "model_id": judge.model_id,
            "raw_model_version": None,
        },
        "status": None,
        "scores": None,
        "error_detail": None,
        "source_code_sha256": source_code_sha256,
        "usage": None,
        "latency_ms": None,
        "evaluated_at": None,
    }


def run_judge(
    judge: Judge,
    rows: list[dict],
    scenarios: tuple[str, ...],
    checkpoint: Checkpoint,
    concurrency: int,
    retry_failed: bool,
) -> None:
    """Runs one judge across every (tool, scenario) pending pair. Payloads (incl. reading
    SOURCE_CODE snippets off disk) are built serially before submission -- local file I/O is
    fast relative to the network call, so there's no need to run it inside the thread pool
    too. Only the slow, I/O-bound `judge.evaluate()` call is concurrent.

    `checkpoint.set()` is only ever called from this (the main) thread, after the JSONL line
    for that task has been written and flushed -- Checkpoint._save() rewrites its whole file
    via a fixed-name temp file on every call, so two threads calling it concurrently could
    race on that file. Scoping the ThreadPoolExecutor strictly around judge.evaluate() avoids
    that without needing to add locking to the already-tested Checkpoint class. The
    flush-then-checkpoint order also bounds the crash-safety cost: if the process dies
    between the two, the worst case on resume is one redundant API call, never a silently
    lost result.

    A failed attempt (technical error or provider quota exhaustion) writes no JSONL line and
    no checkpoint entry -- it leaves zero trace, so a plain re-run (no --retry-failed) retries
    it automatically next round, which matters for unattended/parallel runs (see
    scripts/run_parallel_step3.py) where nobody is watching to pass that flag once a quota
    resets. The trade-off: a permanently-failing (tool, scenario) pair (e.g. a payload that
    always fails to encode) gets retried every round too, burning one API call per round
    until someone notices it repeating in step3_errors_{judge_id}.jsonl and fixes the root
    cause. `--retry-failed` / should_skip()'s "error" branch stay only to unstick "error"
    checkpoint entries written by versions of this script from before this behavior existed.
    """
    out_path = DATA_DIR / "evaluations" / f"{judge.judge_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors_log = LOGS_DIR / f"step3_errors_{judge.judge_id}.jsonl"
    errors_log.parent.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[dict, dict, str]] = []  # (record, payload, checkpoint_key)
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
                tool,
                call_graph,
                repo_src_root_for(name_with_owner),
                name_with_owner,
                include_source=(scenario == "with_source"),
            )
            source_code = payload.get("SOURCE_CODE")
            source_code_sha256 = hashlib.sha256(source_code.encode("utf-8")).hexdigest() if source_code else None
            record = _base_record(row, tool_uid, scenario, judge, source_code_sha256)
            pending.append((record, payload, key))

    if not pending:
        logger.info("[%s] nada a fazer (tudo já concluído)", judge.judge_id)
        return

    logger.info("[%s] %s avaliações pendentes", judge.judge_id, len(pending))
    processed = 0
    recorded = 0
    quota_skipped = 0
    balance_skipped = 0
    generic_skipped = 0
    quota_exhausted = False
    balance_exhausted = False
    with ThreadPoolExecutor(max_workers=concurrency) as pool, open(out_path, "a", encoding="utf-8") as out:
        futures = {pool.submit(judge.evaluate, payload): (record, key) for record, payload, key in pending}
        for future in as_completed(futures):
            if future.cancelled():
                # Never actually ran (cancelled below before the pool got to it) -- no
                # record, no checkpoint entry, so a plain re-run picks it up normally.
                continue

            record, key = futures[future]
            write_result = True
            try:
                result = future.result()
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
            except JudgeRefusal as e:
                record["status"] = "refused"
                record["error_detail"] = f"category={e.category}"
            except JudgeQuotaExhausted as e:
                # Every not-yet-started call would fail identically until the provider's
                # daily quota resets -- cancel the rest of the batch instead of burning
                # through it generating hundreds of duplicate 429s. A task already running
                # when this fires can't be cancelled (ThreadPoolExecutor only cancels queued
                # work); it keeps running and lands here too, like any other result, when
                # its turn in this loop comes up -- only truly-cancelled tasks get skipped
                # (see the `future.cancelled()` check above). Neither kind is written: no
                # JSONL line, no checkpoint entry, so a plain re-run retries all of them
                # once the quota resets, with no manual flag needed.
                write_result = False
                quota_skipped += 1
                if not quota_exhausted:
                    quota_exhausted = True
                    cancelled = sum(1 for f in futures if f.cancel())
                    logger.error(
                        "[%s] cota diária do provedor esgotada -- cancelando o restante desta "
                        "rodada (%s tarefa(s) ainda não iniciadas canceladas). Nenhuma tarefa "
                        "desta rodada (cancelada ou já em andamento) fica marcada como "
                        "processada -- um re-run normal, sem flag, tenta todas de novo depois "
                        "que a cota resetar (~meia-noite Pacific Time).",
                        judge.judge_id, cancelled,
                    )
            except JudgeBalanceExhausted as e:
                # Same cancel-the-rest treatment as JudgeQuotaExhausted above (prepaid
                # balance hit $0 -- every remaining call would fail identically), but this
                # does NOT reset on a schedule: it stays exhausted until a human adds credit,
                # so the log message must say that instead of quoting a reset time.
                write_result = False
                balance_skipped += 1
                if not balance_exhausted:
                    balance_exhausted = True
                    cancelled = sum(1 for f in futures if f.cancel())
                    logger.error(
                        "[%s] saldo do provedor esgotado -- cancelando o restante desta rodada "
                        "(%s tarefa(s) ainda não iniciadas canceladas). Nenhuma tarefa desta "
                        "rodada (cancelada ou já em andamento) fica marcada como processada -- "
                        "um re-run normal, sem flag, tenta todas de novo depois que você "
                        "adicionar crédito na conta (%s).",
                        judge.judge_id, cancelled, e,
                    )
            except Exception as e:  # noqa: BLE001 -- one bad (tool, scenario) must not abort
                # the whole batch, matching clone_all/run_step2's resilience contract. Not
                # written to JSONL/checkpoint either (see docstring) -- only step3_errors_
                # {judge_id}.jsonl keeps a trace, for a human to notice a pair failing every
                # round instead of transiently.
                write_result = False
                generic_skipped += 1
                with open(errors_log, "a", encoding="utf-8") as ef:
                    ef.write(
                        json.dumps(
                            {"tool_uid": record["tool_uid"], "scenario": record["scenario"], "error": str(e)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            if write_result:
                record["evaluated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                checkpoint.set(key, {"status": record["status"]})
                recorded += 1

            processed += 1
            if processed % 20 == 0 or processed == len(pending):
                logger.info("[%s] %s/%s", judge.judge_id, processed, len(pending))

    logger.info(
        "[%s] rodada concluída: %s processadas, %s registradas (ok/refused), %s vão para "
        "retry automático na próxima rodada (%s por cota, %s por saldo, %s por erro técnico)",
        judge.judge_id, processed, recorded,
        quota_skipped + balance_skipped + generic_skipped, quota_skipped, balance_skipped, generic_skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 3: classificação das ferramentas via LLM-as-a-Judge")
    parser.add_argument("--limit", type=int, default=None, help="Avalia só as N primeiras tools do dataset (pilotos/depuração).")
    parser.add_argument("--judges", type=str, default=None, help="IDs de juízes (config/judges.yaml) separados por vírgula. Default: todos com enabled: true.")
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIOS), help=f"Cenários separados por vírgula, dentre {SCENARIOS}.")
    parser.add_argument("--concurrency", type=int, default=5, help="Chamadas concorrentes por juiz.")
    parser.add_argument("--retry-failed", action="store_true", help="Reprocessa também combinações (tool, cenário, juiz) marcadas como erro técnico por uma execução anterior a este script já reprocessar erros automaticamente. Recusas (status=refused) nunca são reprocessadas por esta flag.")
    parser.add_argument("--dataset", type=Path, default=None, help="Caminho para dataset.jsonl (default: data/dataset.jsonl).")
    parser.add_argument("--num-keys", type=int, default=1, help="Total de chaves de API usadas para processar o dataset em paralelo (uma por processo, ver scripts/run_parallel_step3.py). Default 1: processa o dataset inteiro, comportamento normal.")
    parser.add_argument("--key-index", type=int, default=None, help="Índice (0-based) da chave de API deste processo -- processa só a partição de dataset.jsonl correspondente. Obrigatório se --num-keys > 1.")
    args = parser.parse_args()

    scenarios = tuple(args.scenarios.split(","))
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        parser.error(f"cenário(s) desconhecido(s): {sorted(unknown)}. Válidos: {SCENARIOS}")

    if args.num_keys < 1:
        parser.error("--num-keys deve ser >= 1")
    if args.num_keys > 1 and args.key_index is None:
        parser.error("--key-index é obrigatório quando --num-keys > 1")
    if args.key_index is not None and not (0 <= args.key_index < args.num_keys):
        parser.error(f"--key-index deve estar entre 0 e {args.num_keys - 1}")

    ensure_dirs()
    dataset_path = args.dataset or (DATA_DIR / "dataset.jsonl")
    if not dataset_path.exists():
        logger.error("dataset.jsonl não encontrado em %s -- rode a Etapa 2 (assemble_dataset.py) primeiro.", dataset_path)
        sys.exit(1)

    rows = load_dataset_rows(dataset_path, limit=args.limit)
    logger.info("Etapa 3: %s tools carregadas de %s", len(rows), dataset_path)

    if args.num_keys > 1:
        rows = [row for row in rows if shard_for(tool_uid_for(row), args.num_keys) == args.key_index]
        logger.info("Partição da chave %s/%s: %s tools", args.key_index, args.num_keys, len(rows))

    only = set(args.judges.split(",")) if args.judges else None
    judges = load_judges(only=only)
    if not judges:
        logger.error("Nenhum juiz habilitado -- ver config/judges.yaml (enabled: true) ou passe --judges.")
        sys.exit(1)
    logger.info("Juízes: %s", ", ".join(j.judge_id for j in judges))

    # Cada partição usa seu próprio arquivo de checkpoint: Checkpoint._save() reescreve o
    # arquivo inteiro a cada .set(), então dois processos (--key-index diferentes)
    # compartilhando o mesmo arquivo perderiam as escritas um do outro (last-write-wins).
    # O JSONL de saída (data/evaluations/{judge_id}.jsonl) continua compartilhado: cada
    # partição só escreve tool_uids que lhe pertencem, então os appends nunca colidem e não
    # há necessidade de merge depois.
    checkpoint_name = "step3_progress.json" if args.key_index is None else f"step3_progress_key{args.key_index}.json"
    checkpoint = Checkpoint(STATE_DIR / checkpoint_name)

    for judge in judges:
        run_judge(judge, rows, scenarios, checkpoint, args.concurrency, args.retry_failed)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Etapa 3 falhou")
        sys.exit(1)
