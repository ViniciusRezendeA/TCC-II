from __future__ import annotations

import json
from pathlib import Path

from mcp_pipeline.clone.clone_manager import clone_repo
from mcp_pipeline.pipeline.run_step2 import iter_cloned_repos, process_repo
from tests.test_dedupe_rank import make_repo


def _fake_run_success(*args, **kwargs):
    import subprocess

    cmd = args[0]
    if cmd[:2] == ["git", "clone"]:
        Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    if "rev-parse" in cmd:
        return subprocess.CompletedProcess(cmd, 0, stdout="deadbeef1234\n", stderr="")
    raise AssertionError(f"unexpected command: {cmd}")


def _clone_fake_repo(dest_root: Path, name_with_owner: str, language: str, monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setattr("mcp_pipeline.clone.clone_manager.subprocess.run", MagicMock(side_effect=_fake_run_success))
    repo = make_repo(name_with_owner=name_with_owner, language=language)
    return clone_repo(repo, dest_root)


def test_process_repo_detects_python_tools_and_writes_tools_jsonl(tmp_path, monkeypatch):
    dest_root = tmp_path / "repos"
    meta = _clone_fake_repo(dest_root, "acme/weather-mcp", "Python", monkeypatch)
    (meta.src_path / "server.py").write_text(
        '@mcp.tool()\ndef get_weather(city):\n    """Fetch the weather."""\n    return city\n'
    )

    n_tools = process_repo(meta)

    assert n_tools == 1
    tools_file = meta.src_path.parent / "tools.jsonl"
    assert tools_file.exists()
    record = json.loads(tools_file.read_text().strip())
    assert record["tool"]["name"] == "get_weather"
    assert record["call_graph"]["qualified_name"] == "get_weather"


def test_process_repo_with_unsupported_language_writes_empty_file(tmp_path, monkeypatch):
    dest_root = tmp_path / "repos"
    meta = _clone_fake_repo(dest_root, "acme/swift-mcp", "Swift", monkeypatch)

    n_tools = process_repo(meta)

    assert n_tools == 0
    tools_file = meta.src_path.parent / "tools.jsonl"
    assert tools_file.exists()
    assert tools_file.read_text() == ""


def test_process_repo_with_no_tools_writes_empty_file(tmp_path, monkeypatch):
    dest_root = tmp_path / "repos"
    meta = _clone_fake_repo(dest_root, "acme/not-mcp", "Python", monkeypatch)
    (meta.src_path / "readme_gen.py").write_text("def build():\n    return 1\n")

    n_tools = process_repo(meta)

    assert n_tools == 0
    tools_file = meta.src_path.parent / "tools.jsonl"
    assert tools_file.read_text() == ""


def test_process_repo_deletes_source_when_below_min_tools(tmp_path, monkeypatch):
    dest_root = tmp_path / "repos"
    meta = _clone_fake_repo(dest_root, "acme/not-mcp", "Python", monkeypatch)
    (meta.src_path / "readme_gen.py").write_text("def build():\n    return 1\n")

    n_tools = process_repo(meta, min_tools=1)

    assert n_tools == 0
    assert not meta.src_path.exists()
    # repo_meta.json + tools.jsonl (the resumability markers) survive.
    assert meta.src_path.parent.joinpath("repo_meta.json").exists()
    assert meta.src_path.parent.joinpath("tools.jsonl").exists()


def test_process_repo_keeps_source_when_it_meets_min_tools(tmp_path, monkeypatch):
    dest_root = tmp_path / "repos"
    meta = _clone_fake_repo(dest_root, "acme/weather-mcp", "Python", monkeypatch)
    (meta.src_path / "server.py").write_text(
        '@mcp.tool()\ndef get_weather(city):\n    """Fetch the weather."""\n    return city\n'
    )

    n_tools = process_repo(meta, min_tools=1)

    assert n_tools == 1
    assert meta.src_path.exists()


def test_iter_cloned_repos_finds_all_repo_meta_files(tmp_path, monkeypatch):
    dest_root = tmp_path / "repos"
    _clone_fake_repo(dest_root, "acme/one", "Python", monkeypatch)
    _clone_fake_repo(dest_root, "acme/two", "TypeScript", monkeypatch)

    metas = list(iter_cloned_repos(dest_root))

    assert {m.repo.name_with_owner for m in metas} == {"acme/one", "acme/two"}
