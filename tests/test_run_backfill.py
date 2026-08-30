from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from mcp_pipeline.clone.clone_manager import clone_repo
from mcp_pipeline.collection.dedupe_rank import read_jsonl
from mcp_pipeline.pipeline.run_backfill import run_until_target
from mcp_pipeline.pipeline.run_step2 import process_repo
from tests.test_dedupe_rank import make_repo, make_signals


def _fake_run_success(*args, **kwargs):
    import subprocess

    cmd = args[0]
    if cmd[:2] == ["git", "clone"]:
        Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    if "rev-parse" in cmd:
        return subprocess.CompletedProcess(cmd, 0, stdout="deadbeef1234\n", stderr="")
    raise AssertionError(f"unexpected command: {cmd}")


# Real @mcp.tool()-decorated source -- process_repo runs the actual tree-sitter
# detector (not mocked), so a repo only ends up "qualified" if this genuinely
# parses to >= 1 tool, same as test_run_step2.py's fixture.
TOOL_SOURCE = '@mcp.tool()\ndef get_weather(city):\n    """Fetch the weather."""\n    return city\n'
NO_TOOL_SOURCE = "def build():\n    return 1\n"


def _seed_repo(repos_root: Path, repo_id: str, source: str, monkeypatch) -> None:
    monkeypatch.setattr("mcp_pipeline.clone.clone_manager.subprocess.run", MagicMock(side_effect=_fake_run_success))
    repo = make_repo(id=repo_id, name_with_owner=f"acme/{repo_id.lower()}", stars=500, language="Python")
    meta = clone_repo(repo, repos_root)
    (meta.src_path / "server.py").write_text(source)


def test_run_until_target_backfills_from_pool_to_reach_target(tmp_path, monkeypatch):
    repos_root = tmp_path / "repos"
    # R1 (initially selected) turns out to have 0 tools; R2 (initially selected)
    # qualifies; R3 (only in the pool, not initially selected) also qualifies
    # and should get pulled in to cover the resulting deficit of 1.
    _seed_repo(repos_root, "R1", NO_TOOL_SOURCE, monkeypatch)
    _seed_repo(repos_root, "R2", TOOL_SOURCE, monkeypatch)
    _seed_repo(repos_root, "R3", TOOL_SOURCE, monkeypatch)
    _seed_repo(repos_root, "R4", TOOL_SOURCE, monkeypatch)

    pool = [
        make_repo(id="R1", name_with_owner="acme/r1", stars=500),
        make_repo(id="R2", name_with_owner="acme/r2", stars=400),
        make_repo(id="R3", name_with_owner="acme/r3", stars=300),
        make_repo(id="R4", name_with_owner="acme/r4", stars=200),
    ]
    initial_selected = [pool[0], pool[1]]  # R1, R2
    signals = make_signals(top_n=2, min_tools=1)
    selected_path = tmp_path / "selected_repos.jsonl"

    final = run_until_target(
        pool=pool,
        initial_selected=initial_selected,
        repos_root=repos_root,
        signals=signals,
        selected_repos_path=selected_path,
        clone_errors_log=tmp_path / "clone_errors.jsonl",
        step2_errors_log=tmp_path / "step2_errors.jsonl",
    )

    # R1 is disqualified (0 tools) and must be pruned from both the return value
    # and the persisted file -- otherwise a future run would try to clone it again.
    final_ids = {c.id for c in final}
    assert final_ids == {"R2", "R3"}
    assert "R1" not in final_ids
    assert "R4" not in final_ids  # deficit closed after R3, R4 never touched

    persisted_ids = {c.id for c in read_jsonl(selected_path)}
    assert persisted_ids == final_ids


def test_run_until_target_stops_when_pool_exhausted(tmp_path, monkeypatch):
    repos_root = tmp_path / "repos"
    _seed_repo(repos_root, "R1", NO_TOOL_SOURCE, monkeypatch)
    _seed_repo(repos_root, "R2", TOOL_SOURCE, monkeypatch)

    pool = [
        make_repo(id="R1", name_with_owner="acme/r1", stars=500),
        make_repo(id="R2", name_with_owner="acme/r2", stars=400),
    ]
    # top_n=3 but the pool only has 2 candidates total -- can never be satisfied.
    signals = make_signals(top_n=3, min_tools=1)
    selected_path = tmp_path / "selected_repos.jsonl"

    final = run_until_target(
        pool=pool,
        initial_selected=pool,
        repos_root=repos_root,
        signals=signals,
        selected_repos_path=selected_path,
        clone_errors_log=tmp_path / "clone_errors.jsonl",
        step2_errors_log=tmp_path / "step2_errors.jsonl",
    )

    # R1 is disqualified (0 tools), so only R2 survives into the final/persisted set.
    assert {c.id for c in final} == {"R2"}
    assert {c.id for c in read_jsonl(selected_path)} == {"R2"}


def test_run_until_target_wipes_orphaned_clone_dir_and_excludes_it(tmp_path, monkeypatch):
    """Reproduces the real incident: a repo's clone died partway (disk full
    mid-checkout) leaving a directory with no repo_meta.json and no
    tools.jsonl -- clone_all's own is_already_cloned() check can't see this
    as "done", so without this cleanup it would just be retried (and could
    refill the disk) on every future run instead of being recognized as a
    known failure and dropped from the selection.
    """
    repos_root = tmp_path / "repos"
    huge = make_repo(id="HUGE", name_with_owner="acme/huge-repo", stars=999)
    good = make_repo(id="R2", name_with_owner="acme/r2", stars=400)

    orphan_dir = repos_root / "acme__huge-repo" / "src"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "partial_checkout.bin").write_bytes(b"0" * 1024)

    monkeypatch.setattr("mcp_pipeline.clone.clone_manager.subprocess.run", MagicMock(side_effect=_fake_run_success))
    meta_good = clone_repo(good, repos_root)
    (meta_good.src_path / "server.py").write_text(TOOL_SOURCE)

    pool = [huge, good]
    signals = make_signals(top_n=1, min_tools=1)
    selected_path = tmp_path / "selected_repos.jsonl"

    final = run_until_target(
        pool=pool,
        initial_selected=pool,
        repos_root=repos_root,
        signals=signals,
        selected_repos_path=selected_path,
        clone_errors_log=tmp_path / "clone_errors.jsonl",
        step2_errors_log=tmp_path / "step2_errors.jsonl",
    )

    assert {c.id for c in final} == {"R2"}
    assert not (repos_root / "acme__huge-repo").exists()  # orphaned dir wiped
    assert {c.id for c in read_jsonl(selected_path)} == {"R2"}


def test_run_until_target_never_repicks_a_previously_disqualified_repo_from_the_pool(tmp_path, monkeypatch):
    """A repo disqualified in an earlier run (already has tools.jsonl with 0
    tools on disk) and already pruned out of selected_repos.jsonl by that
    prior run must not get re-picked as a "new" backfill candidate on a
    fresh run just because it's still sitting, untouched, in
    candidate_pool.jsonl -- Etapa 1's output is never rewritten, so exclusion
    has to be recomputed from on-disk/log state every time, not just trusted
    from whatever's currently in `initial_selected`.
    """
    repos_root = tmp_path / "repos"
    monkeypatch.setattr("mcp_pipeline.clone.clone_manager.subprocess.run", MagicMock(side_effect=_fake_run_success))

    bad = make_repo(id="BAD", name_with_owner="acme/bad", stars=999)
    r2 = make_repo(id="R2", name_with_owner="acme/r2", stars=400)
    r3 = make_repo(id="R3", name_with_owner="acme/r3", stars=300)

    # Simulate BAD already fully processed and disqualified by a previous run,
    # then pruned out of selected_repos.jsonl -- but still untouched in the pool.
    meta_bad = clone_repo(bad, repos_root)
    (meta_bad.src_path / "server.py").write_text(NO_TOOL_SOURCE)
    process_repo(meta_bad, min_tools=1)

    meta_r2 = clone_repo(r2, repos_root)
    (meta_r2.src_path / "server.py").write_text(TOOL_SOURCE)

    _seed_repo(repos_root, "R3", TOOL_SOURCE, monkeypatch)  # only clones + writes source; not yet processed

    pool = [bad, r2, r3]
    initial_selected = [r2]  # BAD already pruned out on the previous run
    signals = make_signals(top_n=2, min_tools=1)
    selected_path = tmp_path / "selected_repos.jsonl"

    final = run_until_target(
        pool=pool,
        initial_selected=initial_selected,
        repos_root=repos_root,
        signals=signals,
        selected_repos_path=selected_path,
        clone_errors_log=tmp_path / "clone_errors.jsonl",
        step2_errors_log=tmp_path / "step2_errors.jsonl",
    )

    final_ids = {c.id for c in final}
    assert "BAD" not in final_ids
    assert final_ids == {"R2", "R3"}
