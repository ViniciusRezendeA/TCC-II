from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcp_pipeline.clone.clone_manager import (
    CloneError,
    RepoMeta,
    clone_all,
    clone_repo,
    is_already_cloned,
    meta_file_path,
    repo_dir,
    slug_for,
)
from tests.test_dedupe_rank import make_repo


def test_slug_for_replaces_slash_with_double_underscore():
    repo = make_repo(name_with_owner="acme/weather-mcp")
    assert slug_for(repo) == "acme__weather-mcp"


def test_is_already_cloned_false_when_missing(tmp_path):
    repo = make_repo()
    assert is_already_cloned(tmp_path, repo) is False


def test_is_already_cloned_true_when_meta_exists(tmp_path):
    repo = make_repo()
    target = repo_dir(tmp_path, repo)
    target.mkdir(parents=True)
    (target / "repo_meta.json").write_text("{}")
    assert is_already_cloned(tmp_path, repo) is True


def _fake_run_success(*args, **kwargs):
    cmd = args[0]
    if cmd[:2] == ["git", "clone"]:
        # Simulate the clone by creating the destination directory, like real git would.
        dest = cmd[-1]
        Path(dest).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    if cmd[:3] == ["git", "-C", cmd[2]] and "rev-parse" in cmd:
        return subprocess.CompletedProcess(cmd, 0, stdout="deadbeef1234\n", stderr="")
    raise AssertionError(f"unexpected command: {cmd}")


def test_clone_repo_writes_meta_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mcp_pipeline.clone.clone_manager.subprocess.run", MagicMock(side_effect=_fake_run_success)
    )
    repo = make_repo(name_with_owner="acme/weather-mcp")

    meta = clone_repo(repo, tmp_path)

    assert meta.commit_sha == "deadbeef1234"
    meta_file = tmp_path / "acme__weather-mcp" / "repo_meta.json"
    assert meta_file.exists()
    saved = json.loads(meta_file.read_text())
    assert saved["commit_sha"] == "deadbeef1234"
    assert saved["name_with_owner"] == "acme/weather-mcp"


def test_clone_repo_raises_clone_error_on_git_failure(tmp_path, monkeypatch):
    def fake_run_fail(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0], stderr="repository not found")

    monkeypatch.setattr(
        "mcp_pipeline.clone.clone_manager.subprocess.run", MagicMock(side_effect=fake_run_fail)
    )
    repo = make_repo(name_with_owner="acme/does-not-exist")

    with pytest.raises(CloneError):
        clone_repo(repo, tmp_path)


def test_clone_repo_raises_clone_error_when_cleanup_fails(tmp_path, monkeypatch):
    """A filesystem error while clearing a leftover directory (e.g. a stray
    read-only .git object from a killed prior process) must surface as a
    CloneError, not an unhandled OSError, so clone_all's per-repo error
    handling can catch it instead of aborting the whole batch.
    """
    repo = make_repo(name_with_owner="acme/weather-mcp")
    target = repo_dir(tmp_path, repo)
    target.mkdir(parents=True)  # pre-existing leftover directory triggers the rmtree path

    monkeypatch.setattr(
        "mcp_pipeline.clone.clone_manager.shutil.rmtree",
        MagicMock(side_effect=OSError("permission denied")),
    )
    run_mock = MagicMock(side_effect=_fake_run_success)
    monkeypatch.setattr("mcp_pipeline.clone.clone_manager.subprocess.run", run_mock)

    with pytest.raises(CloneError):
        clone_repo(repo, tmp_path)
    run_mock.assert_not_called()


def test_repo_meta_from_meta_file_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mcp_pipeline.clone.clone_manager.subprocess.run", MagicMock(side_effect=_fake_run_success)
    )
    repo = make_repo(name_with_owner="acme/weather-mcp", stars=777)

    original = clone_repo(repo, tmp_path)
    reloaded = RepoMeta.from_meta_file(meta_file_path(tmp_path, repo))

    assert reloaded.repo.name_with_owner == original.repo.name_with_owner
    assert reloaded.repo.stargazer_count == 777
    assert reloaded.commit_sha == original.commit_sha
    assert reloaded.cloned_at == original.cloned_at
    assert reloaded.src_path == original.src_path


def test_clone_all_continues_past_failures_and_logs_them(tmp_path, monkeypatch):
    good = make_repo(id="R_1", name_with_owner="acme/good")
    bad = make_repo(id="R_2", name_with_owner="acme/bad")

    def fake_run(*args, **kwargs):
        cmd = args[0]
        if cmd[:2] == ["git", "clone"] and "acme/bad" in cmd[-2]:
            raise subprocess.CalledProcessError(128, cmd, stderr="not found")
        return _fake_run_success(*args, **kwargs)

    monkeypatch.setattr("mcp_pipeline.clone.clone_manager.subprocess.run", MagicMock(side_effect=fake_run))

    successes, failures = clone_all(
        [good, bad], tmp_path / "repos", tmp_path / "logs" / "clone_errors.jsonl"
    )

    assert len(successes) == 1
    assert successes[0].repo.name_with_owner == "acme/good"
    assert len(failures) == 1
    assert failures[0][0].name_with_owner == "acme/bad"

    errors_log = tmp_path / "logs" / "clone_errors.jsonl"
    assert errors_log.exists()
    logged = json.loads(errors_log.read_text().strip())
    assert logged["repo"] == "acme/bad"


def test_clone_all_includes_already_cloned_repos_in_successes(tmp_path, monkeypatch):
    """A repo cloned in a prior (resumed) run must still show up in
    `successes` on this run — an Etapa 2 orchestrator driving extraction off
    of `successes` must see the complete set, not just newly-cloned repos.
    """
    repo = make_repo(name_with_owner="acme/already-done")
    dest_root = tmp_path / "repos"

    run_mock = MagicMock(side_effect=_fake_run_success)
    monkeypatch.setattr("mcp_pipeline.clone.clone_manager.subprocess.run", run_mock)
    clone_repo(repo, dest_root)  # simulates a prior run having already cloned it
    run_mock.reset_mock()

    successes, failures = clone_all([repo], dest_root, tmp_path / "logs" / "errors.jsonl")

    assert len(successes) == 1
    assert successes[0].repo.name_with_owner == "acme/already-done"
    assert failures == []
    run_mock.assert_not_called()
