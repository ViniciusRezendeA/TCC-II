from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mcp_pipeline.clone.clone_manager import META_FILENAME, RepoMeta
from mcp_pipeline.config import DATA_DIR, LOGS_DIR, ensure_dirs
from mcp_pipeline.extraction.tool_detector import (
    LANGUAGE_ADAPTERS,
    detect_tools_with_call_graphs,
)
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("step2")

TOOLS_FILENAME = "tools.jsonl"


def iter_cloned_repos(repos_root: Path):
    for meta_file in sorted(repos_root.glob(f"*/{META_FILENAME}")):
        yield RepoMeta.from_meta_file(meta_file)


def process_repo(meta: RepoMeta) -> int:
    """Detects tools + builds call graphs for one already-cloned repo,
    writes `tools.jsonl` next to its repo_meta.json. Returns the tool count.
    A repo whose language isn't yet supported (Etapa 2's language coverage
    is incremental — see the plan's Fase 5) or that yields zero real tools
    both legitimately produce an empty tools.jsonl, not an error.
    """
    tools_file = meta.src_path.parent / TOOLS_FILENAME
    language = meta.repo.primary_language

    if language not in LANGUAGE_ADAPTERS:
        tools_file.write_text("", encoding="utf-8")
        logger.info("%s: linguagem %r ainda não suportada, tools.jsonl vazio", meta.repo.name_with_owner, language)
        return 0

    results = detect_tools_with_call_graphs(meta.src_path, language)
    with open(tools_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"tool": tool.to_dict(), "call_graph": graph.to_dict()}, ensure_ascii=False) + "\n" for tool, graph in results)
    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 2: extração de ferramentas e call graph")
    parser.add_argument("--limit", type=int, default=None, help="Processa só os N primeiros repositórios (para pilotos/depuração).")
    parser.add_argument("--retry-failed", action="store_true", help="Reprocessa também repositórios previamente marcados como falha.")
    args = parser.parse_args()

    ensure_dirs()
    repos_root = DATA_DIR / "repos"
    errors_log = LOGS_DIR / "step2_errors.jsonl"
    errors_log.parent.mkdir(parents=True, exist_ok=True)

    previously_failed: set[str] = set()
    if errors_log.exists() and not args.retry_failed:
        for line in errors_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                previously_failed.add(json.loads(line)["repo"])

    all_repos = list(iter_cloned_repos(repos_root))
    if args.limit is not None:
        all_repos = all_repos[: args.limit]

    logger.info("Etapa 2: %s repositórios clonados encontrados em %s", len(all_repos), repos_root)

    processed, skipped, failed = 0, 0, 0
    total_tools = 0
    for i, meta in enumerate(all_repos, 1):
        slug = meta.repo.name_with_owner
        tools_file = meta.src_path.parent / TOOLS_FILENAME

        if tools_file.exists():
            skipped += 1
            continue
        if slug in previously_failed:
            logger.info("[%s/%s] %s falhou antes, pulando (use --retry-failed para tentar de novo)", i, len(all_repos), slug)
            skipped += 1
            continue

        logger.info("[%s/%s] Processando %s (%s)...", i, len(all_repos), slug, meta.repo.primary_language)
        try:
            n_tools = process_repo(meta)
            total_tools += n_tools
            processed += 1
            logger.info("[%s/%s] %s: %s tool(s) encontrada(s)", i, len(all_repos), slug, n_tools)
        except Exception as e:  # noqa: BLE001 — deliberately broad: one malformed/unexpected repo
            # (tree-sitter internals, encoding issues, unanticipated AST shapes) must not abort
            # the other ~205 repos in the batch, matching clone_all's resilience contract.
            failed += 1
            logger.warning("Falha ao processar %s: %s", slug, e)
            with open(errors_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"repo": slug, "error": str(e)}, ensure_ascii=False) + "\n")

    logger.info(
        "Etapa 2 concluída: %s processados, %s pulados (já feitos/falhados), %s falhas nesta rodada, %s tools no total",
        processed, skipped, failed, total_tools,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Etapa 2 falhou")
        sys.exit(1)
