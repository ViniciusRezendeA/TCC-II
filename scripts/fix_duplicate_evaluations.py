#!/usr/bin/env python
"""One-off data repair for two bugs found in data/evaluations/*.jsonl on 2026-09-18
(see JUDGES_STRATEGY.md discussion / run_step3.py::tool_uid_for docstring):

1. tool_uid collision, already fixed going forward in run_step3.py::tool_uid_for
   (commit 3222cbc) but not retroactively: rows written before that commit still
   carry the OLD collided tool_uid for the "lowlevel" SDK patterns, where several
   distinct tools share one handler's source_location. Detected here without any
   dependency on dataset.jsonl: group records by their current tool_uid, and where
   a group holds more than one distinct tool.name, that tool_uid was a collision --
   append the tool name, exactly like the fixed tool_uid_for() does for new runs.
   These are NOT wasted API calls (each row is a real, distinct evaluation) -- just
   mislabeled, and left unfixed they would (a) corrupt the description_only/
   with_source pairing in Etapa 5's analysis (rows for different tools averaged
   together under one key) and (b) make a future run re-evaluate (and, for
   deepseek-flash, re-bill) all of them, since the fixed tool_uid_for() computes a
   different key that the checkpoint has never seen.

2. Genuine duplicate evaluations: same (fixed tool_uid, scenario) evaluated twice
   by the same judge, because two run_step3.py processes didn't share progress --
   either two overlapping single-process runs (deepseek-flash: Checkpoint loads its
   snapshot once at process start and never re-reads the file mid-run) or a
   single-process run and a sharded run_parallel_step3.py run for the same judge
   (gemini-3.5-flash-lite: state/step3_progress.json vs state/step3_progress_key*.json
   are separate files that never see each other's progress). Detected as >1 record
   for the same (fixed tool_uid, scenario) with an IDENTICAL tool.name (i.e. not a
   collision) -- keeps the earliest evaluated_at, drops the rest.

Backs up every file it touches (.bak-<UTC timestamp> alongside the original) before
overwriting, then rebuilds state/step3_progress.json from scratch off the repaired
JSONL files -- that file is the single source of truth checkpoint entries are meant
to mirror, so deriving it fresh (rather than patching it) guarantees the two never
drift apart again. The sharded state/step3_progress_key*.json files are left alone
(out of scope here); running run_parallel_step3.py again for gemini would reproduce
bug 2 unless that script is changed to also consult the main checkpoint.

Idempotent: run again on already-repaired files and it finds nothing to fix.

Usage: uv run python scripts/fix_duplicate_evaluations.py
"""

from __future__ import annotations

import datetime
import json
import shutil
from collections import defaultdict
from pathlib import Path

from mcp_pipeline.config import DATA_DIR, STATE_DIR
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("fix_duplicate_evaluations")

EVALUATION_FILES = [
    DATA_DIR / "evaluations" / "deepseek-flash.jsonl",
    DATA_DIR / "evaluations" / "gemini-3.5-flash-lite.jsonl",
]
CHECKPOINT_PATH = STATE_DIR / "step3_progress.json"


def _backup(path: Path) -> Path:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def repair_file(path: Path) -> list[dict]:
    """Returns the repaired, deduped list of records (also rewrites `path` in place)."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    by_old_uid: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_old_uid[rec["tool_uid"]].append(rec)

    collisions_fixed = 0
    collision_groups = 0
    for old_uid, group in by_old_uid.items():
        names = {r["tool"]["name"] for r in group}
        if len(names) > 1:
            collision_groups += 1
            for rec in group:
                rec["tool_uid"] = f"{old_uid}::{rec['tool']['name']}"
                collisions_fixed += 1

    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        by_pair[(rec["tool_uid"], rec["scenario"])].append(rec)

    kept: list[dict] = []
    dup_pairs = 0
    dup_lines_dropped = 0
    for (_uid, _scenario), group in by_pair.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        dup_pairs += 1
        group.sort(key=lambda r: r["evaluated_at"])
        kept.append(group[0])
        dup_lines_dropped += len(group) - 1

    kept.sort(key=lambda r: r["evaluated_at"])

    logger.info(
        "[%s] %s registros lidos -> %s uid(s) colidida(s) corrigida(s) (%s linhas relabeladas), "
        "%s par(es) duplicado(s) (%s linha(s) removida(s)) -> %s registros finais",
        path.name, len(records), collision_groups, collisions_fixed, dup_pairs, dup_lines_dropped, len(kept),
    )

    if collision_groups == 0 and dup_pairs == 0:
        logger.info("[%s] nada a corrigir, arquivo já está limpo.", path.name)
        return kept

    backup_path = _backup(path)
    logger.info("[%s] backup salvo em %s", path.name, backup_path.name)

    with open(path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return kept


def rebuild_checkpoint(all_records: list[dict]) -> None:
    checkpoint: dict[str, dict] = {}
    for rec in all_records:
        key = f"{rec['tool_uid']}::{rec['scenario']}::{rec['judge']['id']}::{rec['prompt_version']}"
        checkpoint[key] = {"status": rec["status"]}

    if CHECKPOINT_PATH.exists():
        backup_path = _backup(CHECKPOINT_PATH)
        logger.info("checkpoint: backup salvo em %s", backup_path.name)
        old = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    else:
        old = {}

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)

    logger.info(
        "checkpoint reconstruído: %s -> %s chaves (%s removidas, eram uid colidida ou duplicata; "
        "%s são novas -- chaves de linhas que nunca tinham sido checkpointadas)",
        len(old), len(checkpoint),
        len(set(old) - set(checkpoint)), len(set(checkpoint) - set(old)),
    )


def main() -> None:
    all_records: list[dict] = []
    for path in EVALUATION_FILES:
        if not path.exists():
            logger.warning("%s não existe, pulando.", path)
            continue
        all_records.extend(repair_file(path))

    rebuild_checkpoint(all_records)


if __name__ == "__main__":
    main()
