from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from mcp_pipeline.collection.checkpoint import Checkpoint
from mcp_pipeline.config import DATA_DIR, Signals
from mcp_pipeline.github.graphql_client import GraphQLClient, GraphQLError
from mcp_pipeline.github.models import RepoCandidate
from mcp_pipeline.github.queries import REPO_BY_NAME_QUERY, build_manifest_query_string
from mcp_pipeline.github.rest_code_search import cached_page_files, search_code_for_signal

logger = logging.getLogger("mcp_pipeline.search_manifest")

DEFAULT_RAW_PAGES_DIR = DATA_DIR / "raw" / "search_pages"

# Per-repo hydration results (one GraphQL call each), appended to as they
# happen — NOT the same as run_step1.py's RAW_CANDIDATES_PATH, which is only
# written once at the very end of a full run. Hydration has no other
# checkpointing (unlike the REST fetch, which is resumable page by page via
# rest_code_search.py), so without this, any interruption mid-hydration
# (rate limit, a crash, killing a hung run) loses every already-paid-for
# GraphQL call and the next run re-hydrates from scratch — this happened
# three times in a row in one session before this cache was added.
DEFAULT_HYDRATION_CACHE_PATH = DATA_DIR / "raw" / "manifest_hydration_cache.jsonl"


def search_by_manifest_signals(
    client: GraphQLClient,
    checkpoint: Checkpoint,
    signals: Signals,
    github_token: str,
    raw_pages_dir: Path = DEFAULT_RAW_PAGES_DIR,
    hydration_cache_path: Path = DEFAULT_HYDRATION_CACHE_PATH,
) -> Iterator[RepoCandidate]:
    """Finds repos that declare a signal as a library dependency (e.g.
    "@modelcontextprotocol/sdk" in package.json), which the README/description
    text search in search_text.py misses whenever a repo never mentions the
    SDK in prose. Runs one REST Code Search sub-query per (signal, manifest
    file) pair in signals.manifest_signals.

    Resumable page by page for the REST fetch, not just per sub-query:
    search_code_for_signal writes every raw page to disk and checkpoints
    progress before yielding, so an interrupted run (crash, or the 10
    req/min cap tripping a 403) picks up from the last completed page
    instead of losing everything fetched so far for that sub-query.

    Hydration (one GraphQL call per unique repo, to get stars/fork/language)
    is resumable too, via hydration_cache_path — see its module-level
    docstring for why that matters.
    """
    hydration_cache = _load_hydration_cache(hydration_cache_path)
    seen_full_names: set[str] = set()
    for manifest_signal in signals.manifest_signals:
        for file_qualifier in manifest_signal.file_qualifiers:
            label = f"manifest:{manifest_signal.signal}:{file_qualifier}"
            query_string = build_manifest_query_string(manifest_signal.signal, file_qualifier)
            matched_signal = f"manifest:{manifest_signal.signal}"

            full_names = sorted(
                {
                    repo["full_name"]
                    for repo in search_code_for_signal(
                        token=github_token,
                        signal_label=label,
                        query_string=query_string,
                        checkpoint=checkpoint,
                        raw_pages_dir=raw_pages_dir,
                    )
                    if repo.get("full_name")
                }
            )
            logger.info("Manifest sub-query %r matched %s repos", label, len(full_names))

            for full_name in full_names:
                if full_name in seen_full_names:
                    continue
                seen_full_names.add(full_name)

                cached = hydration_cache.get(full_name)
                if cached is not None:
                    yield replace(cached, matched_signals=[matched_signal])
                    continue

                for candidate in _hydrate_repo(client, full_name, matched_signal=matched_signal):
                    _append_to_hydration_cache(hydration_cache_path, candidate)
                    yield candidate


def build_candidates_from_cached_pages(
    signals: Signals, raw_pages_dir: Path = DEFAULT_RAW_PAGES_DIR
) -> list[RepoCandidate]:
    """Builds candidates straight from the already-cached REST code-search
    pages under raw_pages_dir, with zero network calls (no live REST search,
    no GraphQL hydration). Trades data completeness for being instant and
    free: the cached pages only carry what github/rest_code_search.py's
    _trim_repo keeps (id, full_name, owner, description, fork — see its
    docstring), so a RepoCandidate built here has stargazer_count=0,
    primary_language=None, and empty/blank dates — NOT real values, just
    placeholders, since REST code search has no way to report those without
    a follow-up GraphQL call per repo. Only `is_fork` is a real, usable
    signal at this stage, so that's the only filter applied here; there is
    no min_stars cut. Ranked by how many distinct signals matched each repo
    (a repo hit by both an npm and a Python signal is more likely a genuine,
    actively-maintained MCP project than one hit by a single signal), not by
    stars, since stars aren't available without hydrating.

    Only reads raw_pages_dir; never touches the checkpoint file, so it's
    safe to run alongside a real (non-offline) run in progress.
    """
    by_full_name: dict[str, RepoCandidate] = {}
    for manifest_signal in signals.manifest_signals:
        matched_signal = f"manifest:{manifest_signal.signal}"
        for file_qualifier in manifest_signal.file_qualifiers:
            label = f"manifest:{manifest_signal.signal}:{file_qualifier}"
            for page_file in cached_page_files(label, raw_pages_dir):
                payload = json.loads(page_file.read_text(encoding="utf-8"))
                for item in payload.get("items", []):
                    repo = item.get("repository")
                    if not repo or repo.get("fork"):
                        continue
                    full_name = repo.get("full_name")
                    if not full_name:
                        continue
                    existing = by_full_name.get(full_name)
                    if existing is None:
                        by_full_name[full_name] = RepoCandidate(
                            id=repo.get("node_id") or full_name,
                            name_with_owner=full_name,
                            url=repo.get("html_url") or f"https://github.com/{full_name}",
                            description=repo.get("description"),
                            stargazer_count=0,
                            fork_count=0,
                            is_fork=False,
                            is_archived=False,
                            pushed_at="",
                            created_at="",
                            primary_language=None,
                            languages=[],
                            topics=[],
                            default_branch=None,
                            default_branch_oid=None,
                            license_spdx_id=None,
                            matched_signals=[matched_signal],
                        )
                    elif matched_signal not in existing.matched_signals:
                        existing.matched_signals.append(matched_signal)

    candidates = list(by_full_name.values())
    candidates.sort(key=lambda c: (-len(c.matched_signals), c.name_with_owner))
    return candidates


def _load_hydration_cache(path: Path) -> dict[str, RepoCandidate]:
    if not path.exists():
        return {}
    cache: dict[str, RepoCandidate] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidate = RepoCandidate.from_dict(json.loads(line))
                cache[candidate.name_with_owner] = candidate
    return cache


def _append_to_hydration_cache(path: Path, candidate: RepoCandidate) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(candidate.to_dict(), ensure_ascii=False) + "\n")


def _hydrate_repo(client: GraphQLClient, full_name: str, matched_signal: str) -> Iterator[RepoCandidate]:
    owner, _, name = full_name.partition("/")
    try:
        data = client.execute(REPO_BY_NAME_QUERY, {"owner": owner, "name": name})
    except GraphQLError as e:
        # A repo the REST code-search index had can be gone by hydration
        # time (deleted, renamed, or made private) — GitHub's GraphQL API
        # reports that as a NOT_FOUND field error, not `repository: null`,
        # so it surfaces as an exception here rather than reaching the
        # `node is None` check below. Only swallow that specific, expected
        # case; anything else (a real API/query problem) should still fail
        # the run instead of being silently skipped.
        if all(err.get("type") == "NOT_FOUND" for err in e.errors):
            logger.warning("Repo %r from manifest search no longer resolves on GitHub, skipping", full_name)
            return
        raise
    node = data.get("repository")
    if node is None:
        logger.warning("Repo %r from manifest search is no longer accessible, skipping", full_name)
        return
    yield RepoCandidate.from_graphql_node(node, matched_signal=matched_signal)
