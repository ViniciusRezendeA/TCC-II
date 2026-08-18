from __future__ import annotations

from mcp_pipeline.collection.checkpoint import Checkpoint
from mcp_pipeline.github.search_runner import (
    run_paginated_repo_search,
    run_signal_batch,
)


def make_node(id, name_with_owner="acme/repo", stars=500):
    return {
        "id": id,
        "nameWithOwner": name_with_owner,
        "url": f"https://github.com/{name_with_owner}",
        "description": "desc",
        "stargazerCount": stars,
        "forkCount": 0,
        "isFork": False,
        "isArchived": False,
        "pushedAt": "2026-08-01T00:00:00Z",
        "createdAt": "2025-01-01T00:00:00Z",
        "primaryLanguage": {"name": "Python"},
        "languages": {"nodes": [{"name": "Python"}]},
        "repositoryTopics": {"nodes": [{"topic": {"name": "mcp-server"}}]},
        "defaultBranchRef": {"name": "main", "target": {"oid": "abc123"}},
        "licenseInfo": {"spdxId": "MIT"},
    }


class FakeClient:
    """Simulates a two-page GraphQL search response, keyed by the `after` cursor."""

    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def execute(self, query, variables):
        self.calls.append(variables)
        after = variables.get("after")
        page = self.pages[after]
        return {
            "search": {
                "repositoryCount": page["repository_count"],
                "pageInfo": {"hasNextPage": page["has_next"], "endCursor": page["end_cursor"]},
                "nodes": page["nodes"],
            }
        }


def two_page_client(ids=("R_1", "R_2", "R_3")):
    return FakeClient(
        {
            None: {
                "repository_count": 3,
                "has_next": True,
                "end_cursor": "cursor-page-2",
                "nodes": [make_node(ids[0], "acme/one"), make_node(ids[1], "acme/two")],
            },
            "cursor-page-2": {
                "repository_count": 3,
                "has_next": False,
                "end_cursor": "cursor-page-2-end",
                "nodes": [make_node(ids[2], "acme/three")],
            },
        }
    )


def one_page_client(ids=("R_9",)):
    return FakeClient(
        {
            None: {
                "repository_count": 1,
                "has_next": False,
                "end_cursor": "only-page-end",
                "nodes": [make_node(ids[0], "acme/nine")],
            }
        }
    )


QUERY_A = "topic:mcp-server fork:false stars:>=100"
QUERY_B = "topic:mcp-server fork:false stars:>=75"


def test_paginates_through_all_pages(tmp_path):
    client = two_page_client()
    checkpoint = Checkpoint(tmp_path / "state.json")

    results = list(
        run_paginated_repo_search(
            client=client,
            checkpoint=checkpoint,
            signal_label="topic:mcp-server",
            query_string=QUERY_A,
            raw_pages_dir=tmp_path / "raw",
            result_count_warning_threshold=900,
        )
    )

    assert [r.id for r in results] == ["R_1", "R_2", "R_3"]
    assert len(client.calls) == 2


def test_marks_done_in_checkpoint_after_last_page(tmp_path):
    client = two_page_client()
    checkpoint = Checkpoint(tmp_path / "state.json")

    list(
        run_paginated_repo_search(
            client=client,
            checkpoint=checkpoint,
            signal_label="topic:mcp-server",
            query_string=QUERY_A,
            raw_pages_dir=tmp_path / "raw",
            result_count_warning_threshold=900,
        )
    )

    saved = checkpoint.get("search::topic:mcp-server")
    assert saved["done"] is True
    assert saved["query_string"] == QUERY_A


def test_already_completed_signal_replays_from_disk_without_calling_client(tmp_path):
    """A signal marked done must still contribute its repos on a later run
    with the SAME query (e.g. after mcp_signals.yaml changes some other
    signal and dedupe/rank needs to be recomputed) — it should replay its
    previously saved pages, not yield nothing and not re-query.
    """
    raw_dir = tmp_path / "raw"
    first_run_client = two_page_client()
    checkpoint = Checkpoint(tmp_path / "state.json")
    first_results = list(
        run_paginated_repo_search(
            client=first_run_client,
            checkpoint=checkpoint,
            signal_label="topic:mcp-server",
            query_string=QUERY_A,
            raw_pages_dir=raw_dir,
            result_count_warning_threshold=900,
        )
    )
    assert checkpoint.get("search::topic:mcp-server")["done"] is True

    client_that_must_not_be_called = FakeClient({})
    replayed_results = list(
        run_paginated_repo_search(
            client=client_that_must_not_be_called,
            checkpoint=checkpoint,
            signal_label="topic:mcp-server",
            query_string=QUERY_A,
            raw_pages_dir=raw_dir,
            result_count_warning_threshold=900,
        )
    )

    assert [r.id for r in replayed_results] == [r.id for r in first_results]
    assert client_that_must_not_be_called.calls == []


def test_replay_yields_nothing_when_no_cached_pages_exist_but_query_matches(tmp_path):
    checkpoint = Checkpoint(tmp_path / "state.json")
    checkpoint.set("search::topic:mcp-server", {"end_cursor": "x", "done": True, "query_string": QUERY_A})

    results = list(
        run_paginated_repo_search(
            client=FakeClient({}),
            checkpoint=checkpoint,
            signal_label="topic:mcp-server",
            query_string=QUERY_A,
            raw_pages_dir=tmp_path / "raw",
            result_count_warning_threshold=900,
        )
    )

    assert results == []


def test_changed_query_string_is_treated_as_stale_and_requeries(tmp_path):
    """If mcp_signals.yaml changes a query (e.g. min_stars lowered) after a
    signal was already marked done, the old checkpoint must not be trusted —
    replaying the old query's cached pages under a new query_string would
    silently make the config change do nothing.
    """
    raw_dir = tmp_path / "raw"
    checkpoint = Checkpoint(tmp_path / "state.json")

    list(
        run_paginated_repo_search(
            client=two_page_client(),
            checkpoint=checkpoint,
            signal_label="topic:mcp-server",
            query_string=QUERY_A,
            raw_pages_dir=raw_dir,
            result_count_warning_threshold=900,
        )
    )
    assert checkpoint.get("search::topic:mcp-server")["query_string"] == QUERY_A

    new_client = one_page_client(ids=("R_9",))
    results = list(
        run_paginated_repo_search(
            client=new_client,
            checkpoint=checkpoint,
            signal_label="topic:mcp-server",
            query_string=QUERY_B,
            raw_pages_dir=raw_dir,
            result_count_warning_threshold=900,
        )
    )

    assert [r.id for r in results] == ["R_9"]
    assert len(new_client.calls) == 1
    assert checkpoint.get("search::topic:mcp-server")["query_string"] == QUERY_B


def test_changed_query_string_deletes_stale_cached_pages(tmp_path):
    """Stale page files from the old query must not linger and get mixed
    into a future replay of the new query's (possibly shorter) page set.
    """
    raw_dir = tmp_path / "raw"
    checkpoint = Checkpoint(tmp_path / "state.json")

    list(
        run_paginated_repo_search(
            client=two_page_client(),
            checkpoint=checkpoint,
            signal_label="topic:mcp-server",
            query_string=QUERY_A,
            raw_pages_dir=raw_dir,
            result_count_warning_threshold=900,
        )
    )
    assert len(list(raw_dir.glob("*.json"))) == 2

    list(
        run_paginated_repo_search(
            client=one_page_client(),
            checkpoint=checkpoint,
            signal_label="topic:mcp-server",
            query_string=QUERY_B,
            raw_pages_dir=raw_dir,
            result_count_warning_threshold=900,
        )
    )

    remaining_pages = sorted(raw_dir.glob("*.json"))
    assert len(remaining_pages) == 1


def test_writes_raw_pages_to_disk(tmp_path):
    client = two_page_client()
    checkpoint = Checkpoint(tmp_path / "state.json")
    raw_dir = tmp_path / "raw"

    list(
        run_paginated_repo_search(
            client=client,
            checkpoint=checkpoint,
            signal_label="topic:mcp-server",
            query_string=QUERY_A,
            raw_pages_dir=raw_dir,
            result_count_warning_threshold=900,
        )
    )

    saved_pages = sorted(raw_dir.glob("*.json"))
    assert len(saved_pages) == 2


def test_run_signal_batch_iterates_all_labeled_queries(tmp_path):
    raw_dir = tmp_path / "raw"
    checkpoint = Checkpoint(tmp_path / "state.json")
    client = FakeClient(
        {
            None: {
                "repository_count": 1,
                "has_next": False,
                "end_cursor": "end",
                "nodes": [make_node("R_1", "acme/one")],
            }
        }
    )
    # Both signals share the FakeClient's single canned page (fine — we're
    # asserting the batch iterates every label, not that results differ).
    labeled_queries = [("topic:a", "topic:a stars:>=100"), ("topic:b", "topic:b stars:>=100")]

    results = list(
        run_signal_batch(
            client=client,
            checkpoint=checkpoint,
            raw_pages_dir=raw_dir,
            result_count_warning_threshold=900,
            labeled_queries=labeled_queries,
        )
    )

    assert len(results) == 2
    assert checkpoint.get("search::topic:a")["done"] is True
    assert checkpoint.get("search::topic:b")["done"] is True
