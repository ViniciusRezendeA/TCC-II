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

Uso (via -m: importa tool_key_for de scripts.analysis_evaluation_report, então precisa da
raiz do projeto no sys.path -- python scripts/dedupe_evaluations.py direto não resolve isso):
  uv run python -m scripts.dedupe_evaluations                # todos os juízes
  uv run python -m scripts.dedupe_evaluations --judge gemini-3.5-flash-lite
  uv run python -m scripts.dedupe_evaluations --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.logging_setup import setup_logging
from scripts.analysis_evaluation_report import tool_key_for

logger = setup_logging("dedupe_evaluations")

_STATUS_RANK = {"ok": 2, "refused": 2, "error": 1}


def _dedupe_key(record: dict) -> tuple[str, str, str]:
    """tool_key_for() (não o tool_uid bruto do registro) porque tool_uid sozinho colide para
    os padrões de SDK "lowlevel" (python.list_tools_lowlevel, *.set_request_handler_lowlevel):
    várias tools distintas compartilham um tool_uid por herdarem a localização do handler que
    as registra (ver pipeline/run_step3.py::tool_uid_for). Com o tool_uid bruto, este script
    trataria essas tools diferentes como duplicatas umas das outras e apagaria todas menos
    uma -- perda de dado real, não limpeza. Mesma chave usada por
    scripts/generate_dashboard.py (via dedupe_records() abaixo), para as duas ferramentas
    nunca divergirem sobre o que conta como "a mesma avaliação"."""
    return (tool_key_for(record), record["scenario"], record.get("prompt_version", ""))


def _better(candidate: dict, current: dict) -> bool:
    """True if `candidate` should replace `current` as the kept record for a key."""
    candidate_rank = _STATUS_RANK.get(candidate.get("status"), 0)
    current_rank = _STATUS_RANK.get(current.get("status"), 0)
    if candidate_rank != current_rank:
        return candidate_rank > current_rank
    # Same tier (both ok/refused, or both error): keep the most recently evaluated one.
    return (candidate.get("evaluated_at") or "") >= (current.get("evaluated_at") or "")


def dedupe_records(records: list[dict]) -> list[dict]:
    """Dedupe em memória, sem tocar em arquivo -- a mesma seleção de "qual linha vence"
    (ver _better()) usada por dedupe_file() (CLI deste script, reescreve o jsonl) e por
    scripts/generate_dashboard.py (só para montar o dashboard, nunca grava nada). Preserva a
    ordem de primeira aparição de cada chave.
    """
    order: list[tuple[str, str, str]] = []
    kept: dict[tuple[str, str, str], dict] = {}
    for record in records:
        key = _dedupe_key(record)
        if key not in kept:
            order.append(key)
            kept[key] = record
        elif _better(record, kept[key]):
            kept[key] = record
    return [kept[key] for key in order]


def dedupe_file(path: Path, dry_run: bool) -> None:
    if not path.exists():
        logger.warning("%s não existe, pulando", path)
        return

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
        size_at_read = f.tell()

    deduped = dedupe_records(records)
    duplicates_removed = len(records) - len(deduped)
    still_error = sum(1 for r in deduped if r.get("status") == "error")

    logger.info(
        "[%s] linhas: %s -> %s únicas (%s duplicatas removidas); %s ainda com status=error",
        path.name,
        len(records),
        len(deduped),
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
        for record in deduped:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("[%s] reescrito com %s linhas", path.name, len(deduped))


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
