#!/usr/bin/env python
"""Remove avaliações duplicadas de `data/evaluations/{judge_id}.jsonl`.

run_step3.py grava em modo append: uma tool reprocessada via --retry-failed ganha uma
linha NOVA no jsonl, mas a linha antiga (status "error") continua no arquivo -- o
checkpoint (state/step3_progress.json) já reflete só o status mais recente por chave,
mas o jsonl bruto fica com duplicatas por (tool_uid, scenario, prompt_version).

Para cada grupo duplicado, mantém uma única linha, nesta prioridade:
1. A mais recente com status "ok" ou "refused" (avaliação concluída com sucesso/recusa).
2. Se nenhuma teve sucesso, a mais recente com status "error" (nada é descartado
   silenciosamente -- ainda precisa de --retry-failed depois).

Faz backup do arquivo original (.jsonl.bak) antes de sobrescrever. Use --dry-run para
só ver o relatório sem alterar nada.

Uso:
  uv run python scripts/dedupe_evaluations.py                # todos os juízes
  uv run python scripts/dedupe_evaluations.py --judge gemini-3.5-flash-lite
  uv run python scripts/dedupe_evaluations.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("dedupe_evaluations")

_STATUS_RANK = {"ok": 2, "refused": 2, "error": 1}


def _dedupe_key(record: dict) -> tuple[str, str, str]:
    return (record["tool_uid"], record["scenario"], record.get("prompt_version", ""))


def _better(candidate: dict, current: dict) -> bool:
    """True if `candidate` should replace `current` as the kept record for a key."""
    candidate_rank = _STATUS_RANK.get(candidate.get("status"), 0)
    current_rank = _STATUS_RANK.get(current.get("status"), 0)
    if candidate_rank != current_rank:
        return candidate_rank > current_rank
    # Same tier (both ok/refused, or both error): keep the most recently evaluated one.
    return (candidate.get("evaluated_at") or "") >= (current.get("evaluated_at") or "")


def dedupe_file(path: Path, dry_run: bool) -> None:
    if not path.exists():
        logger.warning("%s não existe, pulando", path)
        return

    order: list[tuple[str, str, str]] = []  # first-seen order of keys, for stable output
    kept: dict[tuple[str, str, str], dict] = {}
    total_lines = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            record = json.loads(line)
            key = _dedupe_key(record)
            if key not in kept:
                order.append(key)
                kept[key] = record
            elif _better(record, kept[key]):
                kept[key] = record
        size_at_read = f.tell()

    duplicates_removed = total_lines - len(kept)
    still_error = sum(1 for r in kept.values() if r.get("status") == "error")

    logger.info(
        "[%s] linhas: %s -> %s únicas (%s duplicatas removidas); %s ainda com status=error",
        path.name,
        total_lines,
        len(kept),
        duplicates_removed,
        still_error,
    )

    if duplicates_removed == 0:
        logger.info("[%s] nada a limpar", path.name)
        return

    if dry_run:
        logger.info("[%s] --dry-run: nenhuma alteração gravada", path.name)
        return

    # run_step3.py appends to this same file from a live pipeline run -- if it wrote more
    # bytes after our read finished, rewriting now would silently drop those lines (we'd
    # overwrite with a dedupe computed before they existed). Abort instead of risking data
    # obtained from rate-limited/paid API calls; re-running this script after the pipeline
    # finishes (or is idle) will pick up everything, including the new lines.
    size_now = path.stat().st_size
    if size_now != size_at_read:
        logger.error(
            "[%s] arquivo mudou de tamanho durante a leitura (%s -> %s bytes) -- outro processo "
            "(run_step3.py?) parece estar escrevendo nele agora. Abortando sem gravar para não "
            "perder essas linhas; rode de novo depois que ele terminar.",
            path.name,
            size_at_read,
            size_now,
        )
        return

    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    logger.info("[%s] backup salvo em %s", path.name, backup_path.name)

    with open(path, "w", encoding="utf-8") as out:
        for key in order:
            out.write(json.dumps(kept[key], ensure_ascii=False) + "\n")

    logger.info("[%s] reescrito com %s linhas", path.name, len(kept))


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove avaliações duplicadas dos jsonl de Etapa 3.")
    parser.add_argument("--judge", type=str, default=None, help="judge_id único a limpar (default: todos em data/evaluations/).")
    parser.add_argument("--dry-run", action="store_true", help="Só mostra o relatório, não grava nada.")
    args = parser.parse_args()

    evaluations_dir = DATA_DIR / "evaluations"
    if args.judge:
        paths = [evaluations_dir / f"{args.judge}.jsonl"]
    else:
        paths = sorted(evaluations_dir.glob("*.jsonl"))

    if not paths:
        logger.error("nenhum arquivo de avaliação encontrado em %s", evaluations_dir)
        sys.exit(1)

    for path in paths:
        dedupe_file(path, args.dry_run)


if __name__ == "__main__":
    main()
