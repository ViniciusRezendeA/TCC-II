from __future__ import annotations

from mcp_pipeline.collection.dedupe_rank import dedupe, dedupe_and_rank, filter_and_rank
from mcp_pipeline.config import Signals
from mcp_pipeline.github.models import RepoCandidate


def make_repo(
    id="R_1",
    name_with_owner="acme/weather-mcp",
    stars=500,
    is_fork=False,
    language="Python",
    matched_signal="topic:mcp-server",
) -> RepoCandidate:
    return RepoCandidate(
        id=id,
        name_with_owner=name_with_owner,
        url=f"https://github.com/{name_with_owner}",
        description="A weather MCP server",
        stargazer_count=stars,
        fork_count=0,
        is_fork=is_fork,
        is_archived=False,
        pushed_at="2026-08-01T00:00:00Z",
        created_at="2025-01-01T00:00:00Z",
        primary_language=language,
        languages=[language],
        topics=["mcp-server"],
        default_branch="main",
        default_branch_oid="abc123",
        license_spdx_id="MIT",
        matched_signals=[matched_signal],
    )


def make_signals(**overrides) -> Signals:
    defaults = {
        "topics": ["mcp-server"],
        "text_signals": ["FastMCP"],
        "manifest_signals": [],
        "target_languages": ["Python", "JavaScript", "TypeScript", "Java", "C#"],
        "min_stars": 100,
        "top_n": 206,
        "result_count_warning_threshold": 900,
    }
    defaults.update(overrides)
    return Signals(**defaults)


def test_dedupe_merges_matched_signals_for_same_id():
    a = make_repo(id="R_1", matched_signal="topic:mcp-server")
    b = make_repo(id="R_1", matched_signal="text:FastMCP")
    by_id = dedupe([a, b])
    assert len(by_id) == 1
    assert set(by_id["R_1"].matched_signals) == {"topic:mcp-server", "text:FastMCP"}


def test_dedupe_keeps_distinct_ids_separate():
    a = make_repo(id="R_1")
    b = make_repo(id="R_2", name_with_owner="other/repo")
    by_id = dedupe([a, b])
    assert len(by_id) == 2


def test_filter_and_rank_drops_forks():
    by_id = dedupe([make_repo(id="R_1", is_fork=True)])
    filtered = filter_and_rank(by_id, make_signals())
    assert filtered == []


def test_filter_and_rank_drops_below_min_stars():
    by_id = dedupe([make_repo(id="R_1", stars=50)])
    filtered = filter_and_rank(by_id, make_signals(min_stars=100))
    assert filtered == []


def test_filter_and_rank_keeps_target_language():
    by_id = dedupe([make_repo(id="R_1", language="Python")])
    filtered = filter_and_rank(by_id, make_signals())
    assert [c.id for c in filtered] == ["R_1"]


def test_filter_and_rank_drops_non_target_language():
    by_id = dedupe([make_repo(id="R_1", language="Rust")])
    filtered = filter_and_rank(by_id, make_signals(target_languages=["Python", "JavaScript"]))
    assert filtered == []


def test_filter_and_rank_drops_null_language():
    by_id = dedupe([make_repo(id="R_1", language=None)])
    filtered = filter_and_rank(by_id, make_signals())
    assert filtered == []


def test_filter_and_rank_sorts_by_stars_descending():
    by_id = dedupe(
        [
            make_repo(id="R_1", name_with_owner="a/low", stars=150),
            make_repo(id="R_2", name_with_owner="a/high", stars=900),
            make_repo(id="R_3", name_with_owner="a/mid", stars=400),
        ]
    )
    filtered = filter_and_rank(by_id, make_signals())
    assert [c.stargazer_count for c in filtered] == [900, 400, 150]


def test_dedupe_and_rank_cuts_to_top_n():
    candidates = [
        make_repo(id=f"R_{i}", name_with_owner=f"a/repo{i}", stars=100 + i) for i in range(1, 11)
    ]
    pool, selected = dedupe_and_rank(candidates, make_signals(top_n=3))
    assert len(pool) == 10
    assert len(selected) == 3
    assert [c.stargazer_count for c in selected] == [110, 109, 108]


def test_dedupe_and_rank_handles_pool_smaller_than_top_n(caplog):
    candidates = [make_repo(id="R_1")]
    pool, selected = dedupe_and_rank(candidates, make_signals(top_n=206))
    assert len(pool) == 1
    assert len(selected) == 1
