from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from mcp_pipeline.collection.checkpoint import Checkpoint
from mcp_pipeline.config import DATA_DIR, Signals
from mcp_pipeline.github.graphql_client import GraphQLClient
from mcp_pipeline.github.models import RepoCandidate
from mcp_pipeline.github.queries import build_text_query_string
from mcp_pipeline.github.search_runner import run_signal_batch

DEFAULT_RAW_PAGES_DIR = DATA_DIR / "raw" / "search_pages"


def search_by_text_signals(
    client: GraphQLClient,
    checkpoint: Checkpoint,
    signals: Signals,
    raw_pages_dir: Path = DEFAULT_RAW_PAGES_DIR,
) -> Iterator[RepoCandidate]:
    labeled_queries = (
        (f"text:{signal}", build_text_query_string(signal, signals.min_stars))
        for signal in signals.text_signals
    )
    yield from run_signal_batch(
        client=client,
        checkpoint=checkpoint,
        raw_pages_dir=raw_pages_dir,
        result_count_warning_threshold=signals.result_count_warning_threshold,
        labeled_queries=labeled_queries,
    )
