from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from mcp_pipeline.clone.clone_manager import META_FILENAME, RepoMeta, clone_all, slug_for
from mcp_pipeline.collection.dedupe_rank import read_jsonl, write_jsonl
from mcp_pipeline.config import DATA_DIR, LOGS_DIR, Signals, ensure_dirs
from mcp_pipeline.github.models import RepoCandidate
from mcp_pipeline.logging_setup import setup_logging
from mcp_pipeline.pipeline.run_step2 import TOOLS_FILENAME, process_repo
from mcp_pipeline.schema.assemble_dataset import assemble_dataset

logger = setup_logging("backfill")

DEFAULT_MAX_ITERATIONS = 20


def _repo_status(repos_root: Path, min_tools: int) -> dict[str, str]:
    """id -> 'qualified' | 'disqualified' for every already-processed repo on
    disk (tools.jsonl exists). A repo whose clone or extraction failed never
    gets a tools.jsonl, so it's simply absent here -- callers treat a missing
    id as not-qualified, which is what a permanent failure should count as
    for the deficit math below.
    """
    statuses: dict[str, str] = {}
    for meta_file in repos_root.glob(f"*/{META_FILENAME}"):
        tools_file = meta_file.parent / TOOLS_FILENAME
        if not tools_file.exists():
            continue
        meta = RepoMeta.from_meta_file(meta_file)
        n_tools = sum(1 for line in tools_file.read_text(encoding="utf-8").splitlines() if line.strip())
        statuses[meta.repo.id] = "qualified" if n_tools >= min_tools else "disqualified"
    return statuses


def _load_previously_failed_names(errors_log: Path) -> set[str]:
    if not errors_log.exists():
        return set()
    names: set[str] = set()
    for line in errors_log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            names.add(json.loads(line)["repo"])
    return names


def _orphaned_clone_dirs(repos_root: Path) -> list[Path]:
    """Repo directories that exist on disk but never got a repo_meta.json.
    clone_repo() always mkdir()s the target before attempting the actual git
    clone and only writes repo_meta.json on success -- and now cleans up
    after itself on an ordinary failure too (see clone_manager.py) -- so a
    leftover like this only happens from something clone_repo() can't guard
    against, like the process being killed (or the disk filling up so
    completely that even the failure's own error-log write dies) mid-clone.
    """
    if not repos_root.exists():
        return []
    return [d for d in repos_root.iterdir() if d.is_dir() and not (d / META_FILENAME).exists()]


def _extract_pending(
    repos_root: Path, selected_ids: set[str], failed_ids: set[str], errors_log: Path, min_tools: int
) -> None:
    """Runs Etapa 2 on every selected repo that's cloned but has no
    tools.jsonl yet. Mirrors run_step2.process_repo's own try/except (one bad
    repo can't abort the batch) and writes to the same step2_errors.jsonl, so
    the error log stays a single consistent record regardless of whether
    Etapa 2 ran standalone or via this backfill loop. `min_tools` is passed
    through to process_repo so a disqualified repo's src/ gets cleaned up
    here exactly like it would under run_step2.py directly.
    """
    for meta_file in sorted(repos_root.glob(f"*/{META_FILENAME}")):
        meta = RepoMeta.from_meta_file(meta_file)
        if meta.repo.id not in selected_ids or meta.repo.id in failed_ids:
            continue
        tools_file = meta_file.parent / TOOLS_FILENAME
        if tools_file.exists():
            continue
        try:
            n_tools = process_repo(meta, min_tools=min_tools)
            logger.info("%s: %s tool(s) encontrada(s)", meta.repo.name_with_owner, n_tools)
        except Exception as e:  # noqa: BLE001 -- see run_step2.process_repo's own broad-catch rationale
            failed_ids.add(meta.repo.id)
            logger.warning("Falha ao processar %s: %s", meta.repo.name_with_owner, e)
            with open(errors_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"repo": meta.repo.name_with_owner, "error": str(e)}, ensure_ascii=False) + "\n")


def run_until_target(
    pool: list[RepoCandidate],
    initial_selected: list[RepoCandidate],
    repos_root: Path,
    signals: Signals,
    selected_repos_path: Path,
    clone_errors_log: Path,
    step2_errors_log: Path,
    retry_failed: bool = False,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> list[RepoCandidate]:
    """Clones + extracts the current selection, checks how many cleared
    signals.min_tools, and — if short of signals.top_n — pulls the next-best
    unused candidates from `pool` (already sorted by stars desc, from Etapa
    1's full candidate pool) to fill the gap, then repeats. Stops when the
    deficit closes, the pool has no unused candidates left, or
    max_iterations is hit.

    `selected_repos_path` is rewritten every round, but only with repos that
    are still qualified or not yet resolved — a repo that turns out
    disqualified (< min_tools) or that fails to clone/extract is dropped
    from the persisted file so a future run never tries it again. Exclusion
    is also checked against the *full* `pool` (not just the current
    selection) every run, using persistent, on-disk signals — disqualified
    repos still have their tools.jsonl on disk, and clone/extraction
    failures are logged to `clone_errors_log`/`step2_errors_log` — precisely
    so a repo pruned out of selected_repos.jsonl on a previous run can never
    be re-picked as a "new" backfill candidate later just because it's still
    sitting untouched in candidate_pool.jsonl (Etapa 1's own output, which
    this function never rewrites). An interrupted run still leaves a valid,
    pruned selected_repos.jsonl instead of the stale pre-backfill one.
    """
    pool_by_id = {c.id: c for c in pool}
    selected_ids: list[str] = [c.id for c in initial_selected]

    previously_failed_names = set() if retry_failed else (
        _load_previously_failed_names(clone_errors_log) | _load_previously_failed_names(step2_errors_log)
    )
    on_disk_statuses = _repo_status(repos_root, signals.min_tools)
    failed_ids: set[str] = {
        c.id for c in pool
        if c.name_with_owner in previously_failed_names or on_disk_statuses.get(c.id) == "disqualified"
    }
    used_ids: set[str] = set(selected_ids) | failed_ids

    for orphan_dir in _orphaned_clone_dirs(repos_root):
        matching = next((c for c in pool if slug_for(c) == orphan_dir.name), None)
        size_mb = sum(f.stat().st_size for f in orphan_dir.rglob("*") if f.is_file()) / 1e6
        shutil.rmtree(orphan_dir, ignore_errors=True)
        if matching is None:
            logger.warning(
                "%s: diretório de clone incompleto (sem repo_meta.json, %.1f MB) removido -- repo não está no pool atual",
                orphan_dir.name, size_mb,
            )
            continue
        logger.warning(
            "%s: diretório de clone incompleto (sem repo_meta.json, %.1f MB) removido",
            matching.name_with_owner, size_mb,
        )
        if not retry_failed:
            failed_ids.add(matching.id)
            used_ids.add(matching.id)
            if matching.name_with_owner not in previously_failed_names:
                # The original failure may never have made it into clone_errors_log
                # itself (e.g. disk so full even that write failed -- the exact
                # incident this whole exclusion mechanism exists for), so it's
                # logged here too, on the assumption there's room to write now.
                previously_failed_names.add(matching.name_with_owner)
                with open(clone_errors_log, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {"repo": matching.name_with_owner, "error": "clone incompleto: diretório órfão sem repo_meta.json"},
                            ensure_ascii=False,
                        ) + "\n"
                    )

    persisted: list[RepoCandidate] = []

    for iteration in range(1, max_iterations + 1):
        statuses = _repo_status(repos_root, signals.min_tools)
        excluded_ids = failed_ids | {i for i in selected_ids if statuses.get(i) == "disqualified"}

        active = [pool_by_id[i] for i in selected_ids if i not in excluded_ids]
        _, clone_failures = clone_all(active, repos_root, clone_errors_log)
        for repo, _ in clone_failures:
            failed_ids.add(repo.id)

        _extract_pending(repos_root, set(selected_ids), failed_ids, step2_errors_log, signals.min_tools)

        statuses = _repo_status(repos_root, signals.min_tools)
        excluded_ids = failed_ids | {i for i in selected_ids if statuses.get(i) == "disqualified"}
        persisted = [pool_by_id[i] for i in selected_ids if i not in excluded_ids]
        write_jsonl(persisted, selected_repos_path)

        qualified = sum(1 for i in selected_ids if statuses.get(i) == "qualified")
        deficit = signals.top_n - qualified

        logger.info(
            "Rodada %s: %s/%s selecionados qualificados (>= %s tool(s)), %s excluído(s) "
            "(desqualificados/falhos), déficit=%s",
            iteration, qualified, signals.top_n, signals.min_tools, len(excluded_ids), max(deficit, 0),
        )

        if deficit <= 0:
            break

        remaining_pool = [c for c in pool if c.id not in used_ids]
        backfill = remaining_pool[:deficit]
        if not backfill:
            logger.warning(
                "Pool de candidatos esgotado: só %s/%s selecionados qualificados, sem mais "
                "candidatos livres (pool tem %s no total) para preencher a lacuna. Considere "
                "afrouxar text_signals/min_stars em mcp_signals.yaml e rodar a Etapa 1 de novo.",
                qualified, signals.top_n, len(pool),
            )
            break

        logger.info("Puxando mais %s candidato(s) do pool para preencher a lacuna...", len(backfill))
        for c in backfill:
            selected_ids.append(c.id)
            used_ids.add(c.id)
            if c.name_with_owner in previously_failed_names:
                failed_ids.add(c.id)
    else:
        logger.warning("Limite de %s rodada(s) atingido antes de fechar o déficit.", max_iterations)

    return persisted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clona -> roda Etapa 2 -> checa quantos dos selecionados atingiram min_tools -> "
            "puxa mais candidatos do candidate_pool.jsonl para preencher a lacuna -> repete, "
            "até juntar top_n repositórios qualificados ou esgotar o pool."
        )
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Reprocessa também repositórios previamente marcados como falha na Etapa 2.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"Limite de rodadas do loop (default: {DEFAULT_MAX_ITERATIONS}).",
    )
    args = parser.parse_args()

    ensure_dirs()
    signals = Signals.load()

    pool_path = DATA_DIR / "candidate_pool.jsonl"
    selected_path = DATA_DIR / "selected_repos.jsonl"
    if not pool_path.exists() or not selected_path.exists():
        logger.error("%s / %s não encontrados -- rode a Etapa 1 primeiro.", pool_path, selected_path)
        sys.exit(1)

    final_selected = run_until_target(
        pool=read_jsonl(pool_path),
        initial_selected=read_jsonl(selected_path),
        repos_root=DATA_DIR / "repos",
        signals=signals,
        selected_repos_path=selected_path,
        clone_errors_log=LOGS_DIR / "clone_errors.jsonl",
        step2_errors_log=LOGS_DIR / "step2_errors.jsonl",
        retry_failed=args.retry_failed,
        max_iterations=args.max_iterations,
    )

    count = assemble_dataset(DATA_DIR / "repos", DATA_DIR / "dataset.jsonl", min_tools=signals.min_tools)
    logger.info(
        "Backfill concluído: %s repositório(s) na seleção final, dataset.jsonl com %s tool(s) -> %s",
        len(final_selected), count, DATA_DIR / "dataset.jsonl",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Backfill falhou")
        sys.exit(1)
