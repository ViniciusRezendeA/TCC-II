from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from mcp_pipeline.clone.clone_manager import clone_repo
from mcp_pipeline.pipeline.run_step2 import process_repo
from mcp_pipeline.schema.assemble_dataset import assemble_dataset
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


def test_assemble_dataset_denormalizes_repo_metadata_into_each_row(tmp_path, monkeypatch):
    dest_root = tmp_path / "repos"
    monkeypatch.setattr("mcp_pipeline.clone.clone_manager.subprocess.run", MagicMock(side_effect=_fake_run_success))

    repo = make_repo(name_with_owner="acme/weather-mcp", language="Python", stars=999)
    meta = clone_repo(repo, dest_root)
    (meta.src_path / "server.py").write_text(
        '@mcp.tool()\ndef get_weather(city):\n    """Fetch the weather."""\n    return city\n'
    )
    process_repo(meta)

    output_path = tmp_path / "dataset.jsonl"
    count = assemble_dataset(dest_root, output_path)

    assert count == 1
    row = json.loads(output_path.read_text().strip())
    assert row["schema_version"] == "1.0"
    assert row["repo"]["name_with_owner"] == "acme/weather-mcp"
    assert row["repo"]["stargazer_count"] == 999
    assert row["repo"]["commit_sha"] == "deadbeef1234"
    assert row["tool"]["name"] == "get_weather"
    assert row["call_graph"]["qualified_name"] == "get_weather"
    assert "extracted_at" in row["extraction_meta"]


def test_assemble_dataset_skips_repos_not_yet_processed_by_step2(tmp_path, monkeypatch):
    dest_root = tmp_path / "repos"
    monkeypatch.setattr("mcp_pipeline.clone.clone_manager.subprocess.run", MagicMock(side_effect=_fake_run_success))
    clone_repo(make_repo(name_with_owner="acme/not-processed", language="Python"), dest_root)
    # No tools.jsonl written for this repo — Etapa 2 hasn't run on it yet.

    output_path = tmp_path / "dataset.jsonl"
    count = assemble_dataset(dest_root, output_path)

    assert count == 0
    assert output_path.read_text() == ""


def test_assemble_dataset_produces_zero_rows_for_repos_with_zero_tools(tmp_path, monkeypatch):
    dest_root = tmp_path / "repos"
    monkeypatch.setattr("mcp_pipeline.clone.clone_manager.subprocess.run", MagicMock(side_effect=_fake_run_success))
    repo = make_repo(name_with_owner="acme/not-mcp", language="Python")
    meta = clone_repo(repo, dest_root)
    (meta.src_path / "readme_gen.py").write_text("def build():\n    return 1\n")
    process_repo(meta)  # writes an empty tools.jsonl — legitimate zero-tool outcome

    output_path = tmp_path / "dataset.jsonl"
    count = assemble_dataset(dest_root, output_path)

    assert count == 0
