from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mcp_pipeline.clone.clone_manager import clone_all
from mcp_pipeline.collection.dedupe_rank import read_jsonl
from mcp_pipeline.config import DATA_DIR, LOGS_DIR, ensure_dirs
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("clone")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clona os repositórios selecionados na Etapa 1 (pré-requisito da Etapa 2)")
    parser.add_argument("--limit", type=int, default=None, help="Clona só os N primeiros repositórios (para pilotos/depuração).")
    parser.add_argument("--input", type=Path, default=None, help="Caminho para selected_repos.jsonl (default: data/selected_repos.jsonl).")
    args = parser.parse_args()

    ensure_dirs()
    input_path = args.input or (DATA_DIR / "selected_repos.jsonl")
    repos = read_jsonl(input_path)
    if args.limit is not None:
        repos = repos[: args.limit]

    logger.info("Clonando %s repositórios de %s...", len(repos), input_path)
    successes, failures = clone_all(repos, DATA_DIR / "repos", LOGS_DIR / "clone_errors.jsonl")
    logger.info("Clonagem concluída: %s disponíveis no total, %s falha(s) nesta rodada", len(successes), len(failures))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Clonagem falhou")
        sys.exit(1)
