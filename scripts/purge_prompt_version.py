#!/usr/bin/env python
"""Remove todas as avaliações de um `prompt_version` específico dos jsonl de Etapa 3.

Bumping PROMPT_VERSION (ver prompts.py) nunca apaga resultados antigos -- eles ficam no
jsonl tagueados com sua própria versão, só ficam invisíveis para should_skip() (checkpoint
key inclui prompt_version). Isso é de propósito: permite comparar v2 vs v3 lado a lado
(ver a aba "Versões" em scripts/generate_dashboard.py). Este script é o passo explícito e
manual para de fato descartar uma versão antiga depois que essa comparação não é mais
necessária -- nunca roda como parte do pipeline normal.

Faz backup do arquivo original (.jsonl.bak) antes de sobrescrever. Use --dry-run para só
ver quantas linhas seriam removidas.

Uso:
  uv run python scripts/purge_prompt_version.py --version v2 --dry-run
  uv run python scripts/purge_prompt_version.py --version v2
  uv run python scripts/purge_prompt_version.py --version v2 --judge gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("purge_prompt_version")


def purge_file(path: Path, version: str, dry_run: bool) -> None:
    if not path.exists():
        logger.warning("%s não existe, pulando", path)
        return

    kept: list[str] = []
    removed = 0
    size_at_read = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("prompt_version") == version:
                removed += 1
            else:
                kept.append(stripped)
        size_at_read = f.tell()

    if removed == 0:
        logger.info("[%s] nenhuma linha com prompt_version=%s", path.name, version)
        return

    logger.info("[%s] %s linha(s) com prompt_version=%s de %s totais seriam removidas", path.name, removed, version, removed + len(kept))

    if dry_run:
        logger.info("[%s] --dry-run: nenhuma alteração gravada", path.name)
        return

    # Same concurrent-writer guard as dedupe_evaluations.py: run_step3.py may still be
    # appending to this file.
    size_now = path.stat().st_size
    if size_now != size_at_read:
        logger.error(
            "[%s] arquivo mudou de tamanho durante a leitura -- outro processo parece estar "
            "escrevendo nele agora. Abortando sem gravar; rode de novo depois que ele terminar.",
            path.name,
        )
        return

    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    logger.info("[%s] backup salvo em %s", path.name, backup_path.name)

    with open(path, "w", encoding="utf-8") as out:
        for line in kept:
            out.write(line + "\n")

    logger.info("[%s] reescrito com %s linha(s) restante(s)", path.name, len(kept))


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove todas as avaliações de um prompt_version dos jsonl de Etapa 3.")
    parser.add_argument("--version", type=str, required=True, help="prompt_version a remover, ex.: v2.")
    parser.add_argument("--judge", type=str, default=None, help="judge_id único a limpar (default: todos em data/evaluations/).")
    parser.add_argument("--dry-run", action="store_true", help="Só mostra o relatório, não grava nada.")
    args = parser.parse_args()

    evaluations_dir = DATA_DIR / "evaluations"
    paths = [evaluations_dir / f"{args.judge}.jsonl"] if args.judge else sorted(evaluations_dir.glob("*.jsonl"))

    if not paths:
        logger.error("nenhum arquivo de avaliação encontrado em %s", evaluations_dir)
        sys.exit(1)

    for path in paths:
        purge_file(path, args.version, args.dry_run)


if __name__ == "__main__":
    main()
