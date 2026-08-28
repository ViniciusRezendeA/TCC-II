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
from mcp_pipeline.github.queries import (
    REPO_BY_NAME_QUERY,
    build_manifest_query_string,
)
from mcp_pipeline.github.rest_code_search import (
    cached_page_files,
    search_code_for_signal,
)

logger = logging.getLogger("mcp_pipeline.search_manifest")

DEFAULT_RAW_PAGES_DIR = DATA_DIR / "raw" / "search_pages"

# Per-repo hydration results (one GraphQL call each), appended to as they
# happen — NOT the same as run_step1.py's RAW_CANDIDATES_PATH, which is only
# written once at the very end of a full run.
#
# Hydration has no other checkpointing (unlike the REST fetch, which is
# resumable page by page via rest_code_search.py), so without this, any
# interruption mid-hydration (rate limit, a crash, killing a hung run)
# loses every already-paid-for GraphQL call and the next run re-hydrates
# from scratch.
DEFAULT_HYDRATION_CACHE_PATH = (
    DATA_DIR / "raw" / "manifest_hydration_cache.jsonl"
)


def search_by_manifest_signals(
    client: GraphQLClient,
    checkpoint: Checkpoint,
    signals: Signals,
    github_token: str,
    raw_pages_dir: Path = DEFAULT_RAW_PAGES_DIR,
    hydration_cache_path: Path = DEFAULT_HYDRATION_CACHE_PATH,
) -> Iterator[RepoCandidate]:
    """Find repositories that declare a signal as a library dependency.

    Each manifest signal can define:

    - one or more file qualifiers;
    - one or more target languages.

    A separate REST Code Search sub-query is executed for every
    (signal, manifest file, language) combination.

    For example:

        signal: "@modelcontextprotocol/sdk"
        file_qualifiers:
            - "filename:package.json"
        languages:
            - "JavaScript"
            - "TypeScript"

    produces:

        manifest:@modelcontextprotocol/sdk filename:package.json language:JavaScript
        manifest:@modelcontextprotocol/sdk filename:package.json language:TypeScript

    The language qualifier is pushed into GitHub Code Search itself rather
    than being applied only after the results have been retrieved. This is
    important because GitHub Code Search caps a single query at 1,000
    retrievable results.

    REST fetching is resumable page by page. search_code_for_signal writes
    every raw page to disk and checkpoints progress before yielding, so an
    interrupted run can continue from the last completed page.

    Hydration (one GraphQL call per unique repository, to get stars/fork/
    language and other metadata) is resumable through hydration_cache_path.

    The final star/language validation is intentionally still performed
    downstream by filter_and_rank(). The language qualifier here reduces
    the search population; it does not replace the final client-side
    validation.
    """

    hydration_cache = _load_hydration_cache(hydration_cache_path)

    # Prevent hydrating the same repository more than once during this run,
    # even when it matches multiple manifest signals, file qualifiers, or
    # languages.
    seen_full_names: set[str] = set()

    total_subqueries = sum(
        len(manifest_signal.file_qualifiers) * len(manifest_signal.languages)
        for manifest_signal in signals.manifest_signals
    )
    logger.info(
        "Etapa 1 (busca por manifesto): %s sub-queries planejadas a partir de %s sinais",
        total_subqueries,
        len(signals.manifest_signals),
    )

    subquery_num = 0
    for manifest_signal in signals.manifest_signals:
        for file_qualifier in manifest_signal.file_qualifiers:
            for language in manifest_signal.languages:
                subquery_num += 1
                label = (
                    f"manifest:{manifest_signal.signal}:"
                    f"{file_qualifier}:"
                    f"language={language}"
                )

                query_string = build_manifest_query_string(
                    manifest_signal.signal,
                    file_qualifier,
                    language=language,
                )

                matched_signal = f"manifest:{manifest_signal.signal}"

                logger.info(
                    "[%s/%s] Buscando filtro %r...",
                    subquery_num,
                    total_subqueries,
                    label,
                )

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

                logger.info(
                    "[%s/%s] Filtro %r encontrou %s repos (%s novos após remover já vistos nesta rodada)",
                    subquery_num,
                    total_subqueries,
                    label,
                    len(full_names),
                    len(full_names - seen_full_names),
                )

                for full_name in full_names:
                    if full_name in seen_full_names:
                        continue

                    seen_full_names.add(full_name)

                    cached = hydration_cache.get(full_name)

                    if cached is not None:
                        yield replace(
                            cached,
                            matched_signals=[matched_signal],
                        )
                        continue

                    for candidate in _hydrate_repo(
                        client,
                        full_name,
                        matched_signal=matched_signal,
                    ):
                        _append_to_hydration_cache(
                            hydration_cache_path,
                            candidate,
                        )
                        hydration_cache[
                            candidate.name_with_owner
                        ] = candidate
                        yield candidate


def build_candidates_from_cached_pages(
    signals: Signals,
    raw_pages_dir: Path = DEFAULT_RAW_PAGES_DIR,
) -> list[RepoCandidate]:
    """Build candidates directly from cached REST Code Search pages.

    Performs zero network calls:

    - no live REST search;
    - no GraphQL hydration.

    The cached pages only carry what rest_code_search.py's _trim_repo keeps
    (id, full_name, owner, description, fork, etc.).

    Therefore a RepoCandidate created here has placeholder values for:

    - stargazer_count;
    - primary_language;
    - dates;
    - other GraphQL-only metadata.

    Only is_fork is considered reliable at this stage, so there is no
    min_stars filter here.

    Candidates are ranked by the number of distinct manifest signals that
    matched each repository rather than by stars.

    This function only reads raw_pages_dir and never touches the checkpoint
    file, so it is safe to run alongside a real non-offline run.
    """

    by_full_name: dict[str, RepoCandidate] = {}

    for manifest_signal in signals.manifest_signals:
        matched_signal = f"manifest:{manifest_signal.signal}"

        for file_qualifier in manifest_signal.file_qualifiers:
            for language in manifest_signal.languages:
                label = (
                    f"manifest:{manifest_signal.signal}:"
                    f"{file_qualifier}:"
                    f"language={language}"
                )

                for page_file in cached_page_files(
                    label,
                    raw_pages_dir,
                ):
                    payload = json.loads(
                        page_file.read_text(encoding="utf-8")
                    )

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
                                url=(
                                    repo.get("html_url")
                                    or f"https://github.com/{full_name}"
                                ),
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

                        elif (
                            matched_signal
                            not in existing.matched_signals
                        ):
                            existing.matched_signals.append(
                                matched_signal
                            )

    candidates = list(by_full_name.values())

    candidates.sort(
        key=lambda candidate: (
            -len(candidate.matched_signals),
            candidate.name_with_owner,
        )
    )

    return candidates


def _load_hydration_cache(
    path: Path,
) -> dict[str, RepoCandidate]:
    """Load previously hydrated repositories from the JSONL cache."""

    if not path.exists():
        return {}

    cache: dict[str, RepoCandidate] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            candidate = RepoCandidate.from_dict(json.loads(line))

            cache[candidate.name_with_owner] = candidate

    return cache


def _append_to_hydration_cache(
    path: Path,
    candidate: RepoCandidate,
) -> None:
    """Append one hydrated repository to the JSONL cache."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                candidate.to_dict(),
                ensure_ascii=False,
            )
            + "\n"
        )


def _hydrate_repo(
    client: GraphQLClient,
    full_name: str,
    matched_signal: str,
) -> Iterator[RepoCandidate]:
    """Hydrate one repository using the GraphQL API."""

    owner, _, name = full_name.partition("/")

    try:
        data = client.execute(
            REPO_BY_NAME_QUERY,
            {
                "owner": owner,
                "name": name,
            },
        )

    except GraphQLError as e:
        # A repository found by REST Code Search may no longer exist,
        # have been renamed, deleted, or become private by the time
        # hydration occurs.
        #
        # GitHub GraphQL reports this as a NOT_FOUND field error rather
        # than repository: null.
        #
        # Only swallow this expected case. Any other GraphQL error must
        # still fail the run instead of silently dropping the repository.
        if all(
            err.get("type") == "NOT_FOUND"
            for err in e.errors
        ):
            logger.warning(
                "Repo %r from manifest search no longer resolves "
                "on GitHub, skipping",
                full_name,
            )
            return

        raise

    node = data.get("repository")

    if node is None:
        logger.warning(
            "Repo %r from manifest search is no longer accessible, skipping",
            full_name,
        )
        return

    yield RepoCandidate.from_graphql_node(
        node,
        matched_signal=matched_signal,
    )
