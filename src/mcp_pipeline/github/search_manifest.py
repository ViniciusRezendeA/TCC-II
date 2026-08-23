from __future__ import annotations

import logging
from collections.abc import Iterator

from mcp_pipeline.collection.checkpoint import Checkpoint
from mcp_pipeline.config import Signals
from mcp_pipeline.github.graphql_client import GraphQLClient
from mcp_pipeline.github.models import RepoCandidate
from mcp_pipeline.github.queries import REPO_BY_NAME_QUERY, build_manifest_query_string
from mcp_pipeline.github.rest_code_search import search_code_for_signal

logger = logging.getLogger("mcp_pipeline.search_manifest")


def search_by_manifest_signals(
    client: GraphQLClient,
    checkpoint: Checkpoint,
    signals: Signals,
    github_token: str,
) -> Iterator[RepoCandidate]:
    """Finds repos that declare a signal as a library dependency (e.g.
    "@modelcontextprotocol/sdk" in package.json), which the README/description
    text search in search_text.py misses whenever a repo never mentions the
    SDK in prose. Runs one REST Code Search sub-query per (signal, manifest
    file) pair in signals.manifest_signals.

    Resumable per sub-query (like search_text.py/search_topics.py), but not
    per-page within a sub-query — REST Code Search is capped at 10 req/min,
    so a sub-query's page count is small enough that redoing it after a crash
    is cheap, and the checkpoint here exists mainly to skip sub-queries
    that already completed in a previous run of the whole pipeline.
    """
    seen_full_names: set[str] = set()
    for manifest_signal in signals.manifest_signals:
        for file_qualifier in manifest_signal.file_qualifiers:
            label = f"manifest:{manifest_signal.signal}:{file_qualifier}"
            query_string = build_manifest_query_string(manifest_signal.signal, file_qualifier)
            state_key = f"search::{label}"
            saved = checkpoint.get(state_key, {})

            if saved.get("done") and saved.get("query_string") == query_string:
                logger.info("Manifest sub-query %r already completed, skipping", label)
                continue

            full_names = sorted(
                {
                    repo["full_name"]
                    for repo in search_code_for_signal(github_token, query_string)
                    if repo.get("full_name")
                }
            )
            checkpoint.set(state_key, {"done": True, "query_string": query_string})
            logger.info("Manifest sub-query %r matched %s repos", label, len(full_names))

            for full_name in full_names:
                if full_name in seen_full_names:
                    continue
                seen_full_names.add(full_name)
                yield from _hydrate_repo(client, full_name, matched_signal=f"manifest:{manifest_signal.signal}")


def _hydrate_repo(client: GraphQLClient, full_name: str, matched_signal: str) -> Iterator[RepoCandidate]:
    owner, _, name = full_name.partition("/")
    data = client.execute(REPO_BY_NAME_QUERY, {"owner": owner, "name": name})
    node = data.get("repository")
    if node is None:
        logger.warning("Repo %r from manifest search is no longer accessible, skipping", full_name)
        return
    yield RepoCandidate.from_graphql_node(node, matched_signal=matched_signal)
