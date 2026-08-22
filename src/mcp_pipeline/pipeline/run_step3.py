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
from mcp_pipeline.evaluation.judges.base import Judge, JudgeRefusal
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
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool, open(out_path, "a", encoding="utf-8") as out:
        futures = {pool.submit(judge.evaluate, payload): (record, key) for record, payload, key in pending}
        for future in as_completed(futures):
            record, key = futures[future]
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
            except Exception as e:  # noqa: BLE001 -- one bad (tool, scenario) must not abort
                # the whole batch, matching clone_all/run_step2's resilience contract.
                record["status"] = "error"
                record["error_detail"] = str(e)
                with open(errors_log, "a", encoding="utf-8") as ef:
                    ef.write(
                        json.dumps(
                            {"tool_uid": record["tool_uid"], "scenario": record["scenario"], "error": str(e)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            record["evaluated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            checkpoint.set(key, {"status": record["status"]})

            done += 1
            if done % 20 == 0 or done == len(pending):
                logger.info("[%s] %s/%s", judge.judge_id, done, len(pending))

    logger.info("[%s] concluído: %s avaliações nesta rodada", judge.judge_id, done)


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 3: classificação das ferramentas via LLM-as-a-Judge")
    parser.add_argument("--limit", type=int, default=None, help="Avalia só as N primeiras tools do dataset (pilotos/depuração).")
    parser.add_argument("--judges", type=str, default=None, help="IDs de juízes (config/judges.yaml) separados por vírgula. Default: todos com enabled: true.")
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIOS), help=f"Cenários separados por vírgula, dentre {SCENARIOS}.")
    parser.add_argument("--concurrency", type=int, default=5, help="Chamadas concorrentes por juiz.")
    parser.add_argument("--retry-failed", action="store_true", help="Reprocessa também combinações (tool, cenário, juiz) previamente marcadas como erro técnico. Recusas (status=refused) nunca são reprocessadas por esta flag.")
    parser.add_argument("--dataset", type=Path, default=None, help="Caminho para dataset.jsonl (default: data/dataset.jsonl).")
    args = parser.parse_args()

    scenarios = tuple(args.scenarios.split(","))
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        parser.error(f"cenário(s) desconhecido(s): {sorted(unknown)}. Válidos: {SCENARIOS}")

    ensure_dirs()
    dataset_path = args.dataset or (DATA_DIR / "dataset.jsonl")
    if not dataset_path.exists():
        logger.error("dataset.jsonl não encontrado em %s -- rode a Etapa 2 (assemble_dataset.py) primeiro.", dataset_path)
        sys.exit(1)

    rows = load_dataset_rows(dataset_path, limit=args.limit)
    logger.info("Etapa 3: %s tools carregadas de %s", len(rows), dataset_path)

    only = set(args.judges.split(",")) if args.judges else None
    judges = load_judges(only=only)
    if not judges:
        logger.error("Nenhum juiz habilitado -- ver config/judges.yaml (enabled: true) ou passe --judges.")
        sys.exit(1)
    logger.info("Juízes: %s", ", ".join(j.judge_id for j in judges))

    checkpoint = Checkpoint(STATE_DIR / "step3_progress.json")

    for judge in judges:
        run_judge(judge, rows, scenarios, checkpoint, args.concurrency, args.retry_failed)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Etapa 3 falhou")
        sys.exit(1)
