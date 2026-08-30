from __future__ import annotations

"""Retroativamente remove o código-fonte clonado (src/) de repositórios já
processados pela Etapa 2 que ficaram abaixo de min_tools (mcp_signals.yaml).

process_repo() (run_step2.py) já faz essa limpeza sozinho para repos
processados a partir de agora -- este script cobre repos processados antes
dessa mudança existir, ou depois de ajustar min_tools no config. Só apaga
src/; repo_meta.json e tools.jsonl (os marcadores que clone_all/process_repo
usam para pular repos já feitos) nunca são tocados, então reprocessar depois
continua funcionando normalmente.

Uso:
    uv run python -m scripts.cleanup_disqualified_repos [--dry-run]
"""

import argparse
import shutil
from pathlib import Path

from mcp_pipeline.clone.clone_manager import META_FILENAME, RepoMeta
from mcp_pipeline.config import DATA_DIR, Signals
from mcp_pipeline.logging_setup import setup_logging
from mcp_pipeline.pipeline.run_step2 import TOOLS_FILENAME

logger = setup_logging("cleanup_disqualified_repos")


def cleanup(repos_root: Path, min_tools: int, dry_run: bool = False) -> tuple[int, int]:
    """Returns (n_repos_cleaned, bytes_freed)."""
    n_cleaned = 0
    bytes_freed = 0
    for meta_file in sorted(repos_root.glob(f"*/{META_FILENAME}")):
        tools_file = meta_file.parent / TOOLS_FILENAME
        src_path = meta_file.parent / "src"
        if not tools_file.exists() or not src_path.exists():
            continue

        n_tools = sum(1 for line in tools_file.read_text(encoding="utf-8").splitlines() if line.strip())
        if n_tools >= min_tools:
            continue

        meta = RepoMeta.from_meta_file(meta_file)
        size = sum(f.stat().st_size for f in src_path.rglob("*") if f.is_file())
        verb = "removeria" if dry_run else "removidos"
        logger.info(
            "%s%s: %s tool(s) < min_tools=%s, %s %.1f MB",
            "[dry-run] " if dry_run else "", meta.repo.name_with_owner, n_tools, min_tools, verb, size / 1e6,
        )
        if not dry_run:
            shutil.rmtree(src_path)

        n_cleaned += 1
        bytes_freed += size
    return n_cleaned, bytes_freed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove retroativamente o src/ de repos já processados que ficaram abaixo de min_tools."
    )
    parser.add_argument("--dry-run", action="store_true", help="Só mostra o que seria removido, sem apagar nada.")
    args = parser.parse_args()

    signals = Signals.load()
    n_cleaned, bytes_freed = cleanup(DATA_DIR / "repos", signals.min_tools, dry_run=args.dry_run)
    logger.info(
        "%s%s repo(s) %s, %.2f GB %s",
        "[dry-run] " if args.dry_run else "",
        n_cleaned,
        "seriam limpos" if args.dry_run else "limpos",
        bytes_freed / 1e9,
        "seriam liberados" if args.dry_run else "liberados",
    )


if __name__ == "__main__":
    main()
