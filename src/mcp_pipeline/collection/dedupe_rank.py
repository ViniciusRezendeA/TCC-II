from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from mcp_pipeline.config import Signals
from mcp_pipeline.github.models import RepoCandidate

logger = logging.getLogger("mcp_pipeline.dedupe_rank")


def dedupe(candidates: Iterable[RepoCandidate]) -> dict[str, RepoCandidate]:
    """Unions candidates by GraphQL node `id` (stable across renames, unlike
    nameWithOwner). When the same repo is matched by more than one
    topic/text-signal query, its `matched_signals` list accumulates every
    signal that found it, instead of keeping only the first.
    """
    by_id: dict[str, RepoCandidate] = {}
    for candidate in candidates:
        existing = by_id.get(candidate.id)
        if existing is None:
            by_id[candidate.id] = candidate
        else:
            for signal in candidate.matched_signals:
                if signal not in existing.matched_signals:
                    existing.matched_signals.append(signal)

    return by_id


def filter_and_rank(
    by_id: dict[str, RepoCandidate], signals: Signals
) -> list[RepoCandidate]:
    """Belt-and-suspenders re-validation of the fork/star filters already
    inlined in the GraphQL query strings (and applied client-side for the
    REST manifest source, which has no working fork:/stars: qualifier — see
    queries.py). Also drops repos whose GitHub-detected primaryLanguage isn't
    in signals.target_languages, so the top_n slots aren't spent on languages
    Etapa 2 has no (and no planned) extraction support for (e.g. HTML,
    Jupyter Notebook, Dockerfile, Shell). Returns the full filtered pool
    sorted by stars descending — NOT yet cut down to top_n.
    """
    filtered = [
        c
        for c in by_id.values()
        if not c.is_fork
        and c.stargazer_count >= signals.min_stars
        and c.primary_language in signals.target_languages
    ]
    filtered.sort(key=lambda c: c.stargazer_count, reverse=True)
    return filtered


def write_jsonl(candidates: list[RepoCandidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in candidates)


def read_jsonl(path: Path) -> list[RepoCandidate]:
    candidates = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(RepoCandidate.from_dict(json.loads(line)))
    return candidates


def dedupe_and_rank(
    candidates: Iterable[RepoCandidate], signals: Signals
) -> tuple[list[RepoCandidate], list[RepoCandidate]]:
    """Returns (full_candidate_pool, top_n_selected), both sorted by stars desc."""
    by_id = dedupe(candidates)
    logger.info("Deduped to %s unique repository ids", len(by_id))

    pool = filter_and_rank(by_id, signals)
    logger.info(
        "%s repos pass fork/star filters (of %s unique candidates)",
        len(pool),
        len(by_id),
    )

    selected = pool[: signals.top_n]
    if len(selected) < signals.top_n:
        logger.warning(
            "Candidate pool (%s) is smaller than top_n target (%s) — see plan's "
            "contingency policy: broaden text_signals in mcp_signals.yaml, then "
            "consider lowering min_stars, before reporting a smaller N.",
            len(selected),
            signals.top_n,
        )
    return pool, selected
