from __future__ import annotations

from mcp_pipeline.collection.checkpoint import Checkpoint


def test_get_returns_default_when_missing(tmp_path):
    cp = Checkpoint(tmp_path / "state.json")
    assert cp.get("missing_key", "default") == "default"


def test_set_then_get_round_trips(tmp_path):
    cp = Checkpoint(tmp_path / "state.json")
    cp.set("topic:mcp-server", {"end_cursor": "abc", "done": False})
    assert cp.get("topic:mcp-server") == {"end_cursor": "abc", "done": False}


def test_persists_across_instances(tmp_path):
    path = tmp_path / "state.json"
    Checkpoint(path).set("key", "value")
    reloaded = Checkpoint(path)
    assert reloaded.get("key") == "value"


def test_all_returns_full_snapshot(tmp_path):
    cp = Checkpoint(tmp_path / "state.json")
    cp.set("a", 1)
    cp.set("b", 2)
    assert cp.all() == {"a": 1, "b": 2}
