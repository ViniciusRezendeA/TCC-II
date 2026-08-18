from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from mcp_pipeline.collection.checkpoint import Checkpoint
from mcp_pipeline.github.graphql_client import GraphQLClient
from mcp_pipeline.github.models import RepoCandidate
from mcp_pipeline.github.queries import SEARCH_REPOS_QUERY

logger = logging.getLogger("mcp_pipeline.search_runner")


def run_signal_batch(
    client: GraphQLClient,
    checkpoint: Checkpoint,
    raw_pages_dir: Path,
    result_count_warning_threshold: int,
    labeled_queries: Iterable[tuple[str, str]],
) -> Iterator[RepoCandidate]:
    """Runs `run_paginated_repo_search` for each (signal_label, query_string)
    pair in turn. Shared by search_topics.py and search_text.py (and any
    future "iterate over N labeled query variants" caller, e.g. Etapa 2's
    per-language/per-SDK-generation detectors) so the iteration shape is
    defined once instead of copy-pasted per caller.
    """
    for signal_label, query_string in labeled_queries:
        yield from run_paginated_repo_search(
            client=client,
            checkpoint=checkpoint,
            signal_label=signal_label,
            query_string=query_string,
            raw_pages_dir=raw_pages_dir,
            result_count_warning_threshold=result_count_warning_threshold,
        )


def run_paginated_repo_search(
    client: GraphQLClient,
    checkpoint: Checkpoint,
    signal_label: str,
    query_string: str,
    raw_pages_dir: Path,
    result_count_warning_threshold: int,
) -> Iterator[RepoCandidate]:
    """Runs one GitHub repository-search query to exhaustion, yielding
    RepoCandidates page by page. Resumable: if `signal_label` already has a
    saved cursor in `checkpoint` from a previous (interrupted) run, resumes
    from there instead of restarting.

    Every raw page is written to disk under `raw_pages_dir` before being
    yielded, so a crash never loses an already-paid-for API call.

    If `signal_label` is already marked done AND `query_string` matches what
    produced that saved state, replays the previously saved raw pages from
    disk instead of re-querying — a "done" signal still has to contribute
    its repos to every subsequent run (e.g. after the signal set in
    mcp_signals.yaml changes and dedupe/rank needs to be recomputed), not
    just the run that originally fetched it.

    If `query_string` has changed since the saved "done" state (e.g.
    min_stars was lowered in mcp_signals.yaml), the saved state is treated
    as stale and the query is re-run from scratch rather than silently
    replaying results for a different query — a stale checkpoint that's
    trusted blindly would otherwise make a config change silently do
    nothing.
    """
    state_key = f"search::{signal_label}"
    saved = checkpoint.get(state_key, {})
    done = saved.get("done", False)
    saved_query_string = saved.get("query_string")

    if done and saved_query_string == query_string:
        logger.info("Sub-query %r already completed in a previous run, replaying cached pages", signal_label)
        yield from _replay_cached_pages(signal_label, raw_pages_dir)
        return

    if done and saved_query_string != query_string:
        logger.warning(
            "Sub-query %r was marked done for a different query string (%r vs current %r) "
            "— treating checkpoint as stale and re-running from scratch",
            signal_label,
            saved_query_string,
            query_string,
        )
        _delete_cached_pages(signal_label, raw_pages_dir)
        after = None
        page_num = 0
    else:
        after = saved.get("end_cursor")
        page_num = saved.get("next_page_num", 0)

    warned_broad = False

    while True:
        data = client.execute(SEARCH_REPOS_QUERY, {"queryString": query_string, "after": after})
        search = data["search"]
        repo_count = search["repositoryCount"]

        if not warned_broad and repo_count >= result_count_warning_threshold:
            logger.warning(
                "Sub-query %r matched %s repos (approaching the 1000-result GitHub "
                "Search cap) — signal may be too broad",
                signal_label,
                repo_count,
            )
            warned_broad = True

        raw_pages_dir.mkdir(parents=True, exist_ok=True)
        page_file = raw_pages_dir / f"{_safe_filename(signal_label)}__page{page_num:04d}.json"
        page_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        for node in search["nodes"]:
            yield RepoCandidate.from_graphql_node(node, matched_signal=signal_label)

        page_info = search["pageInfo"]
        after = page_info["endCursor"]
        page_num += 1
        has_next = page_info["hasNextPage"]

        checkpoint.set(
            state_key,
            {
                "end_cursor": after,
                "next_page_num": page_num,
                "done": not has_next,
                "query_string": query_string,
            },
        )

        if not has_next:
            logger.info(
                "Sub-query %r complete: %s total repos matched across %s page(s)",
                signal_label,
                repo_count,
                page_num,
            )
            break


def _replay_cached_pages(signal_label: str, raw_pages_dir: Path) -> Iterator[RepoCandidate]:
    for page_file in _cached_page_files(signal_label, raw_pages_dir):
        data = json.loads(page_file.read_text(encoding="utf-8"))
        for node in data["search"]["nodes"]:
            yield RepoCandidate.from_graphql_node(node, matched_signal=signal_label)


def _delete_cached_pages(signal_label: str, raw_pages_dir: Path) -> None:
    """Removes stale page files left over from a query_string that no
    longer matches, so a future replay never mixes pages from two different
    queries (e.g. leftover page0005.json from a broader old query, past the
    page count a narrower new query actually produces).
    """
    for page_file in _cached_page_files(signal_label, raw_pages_dir):
        page_file.unlink()


def _cached_page_files(signal_label: str, raw_pages_dir: Path) -> list[Path]:
    prefix = f"{_safe_filename(signal_label)}__page"
    return sorted(raw_pages_dir.glob(f"{prefix}*.json"))


def _safe_filename(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
