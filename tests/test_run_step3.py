from __future__ import annotations

import json
import time

from mcp_pipeline.collection.checkpoint import Checkpoint
from mcp_pipeline.evaluation.judges.base import (
    JudgeError,
    JudgeEvaluation,
    JudgeQuotaExhausted,
    RubricScores,
)
from mcp_pipeline.pipeline.run_step3 import (
    checkpoint_key,
    run_judge,
    shard_for,
    should_skip,
    tool_uid_for,
)


def _rubric_scores() -> RubricScores:
    component = {"score": 4, "reasoning": "ok"}
    return RubricScores.model_validate(
        {
            "purpose": component,
            "guidelines": component,
            "limitations": component,
            "parameter_explanation": component,
            "length_completeness": component,
            "examples": component,
        }
    )


class FakeJudge:
    """A Judge implementation with no real SDK behind it, for exercising run_judge's
    checkpointing/resumability/error-handling without mocking a provider client.
    """

    def __init__(self, judge_id: str = "fake-judge", fail_on: set[str] | None = None):
        self.judge_id = judge_id
        self.provider = "fake"
        self.model_id = "fake-model"
        self.calls: list[dict] = []
        self._fail_on = fail_on or set()

    def evaluate(self, payload: dict) -> JudgeEvaluation:
        self.calls.append(payload)
        if payload["name"] in self._fail_on:
            raise JudgeError("simulated failure")
        return JudgeEvaluation(
            scores=_rubric_scores(),
            input_tokens=10,
            output_tokens=20,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw_model_version="fake-model-v1",
            latency_ms=1.0,
        )


def _make_row(name: str = "get_weather", file: str = "server.py", start_line: int = 1) -> dict:
    return {
        "repo": {"name_with_owner": "acme/weather-mcp", "stargazer_count": 10, "primary_language": "Python"},
        "tool": {
            "name": name,
            "qualified_name": name,
            "description": "Fetch the weather.",
            "description_is_literal": True,
            "sdk_pattern": "python.fastmcp_decorator",
            "source_location": {"file": file, "start_line": start_line, "end_line": start_line + 1},
            "loc": 2,
            "call_graph_depth": 1,
            "cyclomatic_complexity": 1,
        },
        "call_graph": {
            "level": 1,
            "resolved": True,
            "external": False,
            "ambiguous": False,
            "qualified_name": name,
            "raw_call_text": None,
            "source_location": {"file": file, "start_line": start_line, "end_line": start_line + 1},
            "calls": [],
        },
    }


def _isolate_data_dirs(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr("mcp_pipeline.pipeline.run_step3.DATA_DIR", data_dir)
    monkeypatch.setattr("mcp_pipeline.pipeline.run_step3.LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr("mcp_pipeline.evaluation.payload.DATA_DIR", data_dir)
    repo_src = data_dir / "repos" / "acme__weather-mcp" / "src"
    repo_src.mkdir(parents=True)
    (repo_src / "server.py").write_text("def get_weather(city):\n    return city\n")
    return data_dir


def test_tool_uid_for_is_stable_and_includes_source_location():
    row = _make_row()

    assert tool_uid_for(row) == "acme/weather-mcp::get_weather::server.py:1"


def test_tool_uid_for_disambiguates_lowlevel_tools_sharing_one_handler():
    """python.list_tools_lowlevel/*.set_request_handler_lowlevel put N distinct tools at the
    same qualified_name + source_location (the shared handler's own location) -- tool_uid_for
    must fold in tool.name for these patterns, or two different tools collide onto one uid.
    """
    row_a = _make_row(name="search_models")
    row_a["tool"]["sdk_pattern"] = "python.list_tools_lowlevel"
    row_a["tool"]["qualified_name"] = "handle_list_tools"
    row_b = _make_row(name="search_datasets")
    row_b["tool"]["sdk_pattern"] = "python.list_tools_lowlevel"
    row_b["tool"]["qualified_name"] = "handle_list_tools"

    assert tool_uid_for(row_a) != tool_uid_for(row_b)
    assert tool_uid_for(row_a) == "acme/weather-mcp::handle_list_tools::server.py:1::search_models"


def test_tool_uid_for_unaffected_for_non_lowlevel_patterns():
    """Every other sdk_pattern already has a per-tool source_location -- tool_uid must stay
    byte-identical to before the lowlevel fix, so existing checkpoints/evaluations for the
    ~70% of the dataset outside the lowlevel patterns aren't invalidated by it."""
    row = _make_row()

    assert row["tool"]["sdk_pattern"] == "python.fastmcp_decorator"
    assert tool_uid_for(row) == "acme/weather-mcp::get_weather::server.py:1"


def test_checkpoint_key_embeds_prompt_version():
    from mcp_pipeline.evaluation.prompts import PROMPT_VERSION

    key = checkpoint_key("acme/x::t::f.py:1", "description_only", "qwen2.5-14b-instruct")

    assert key.endswith(f"::{PROMPT_VERSION}")


def test_shard_for_is_deterministic():
    uid = "acme/weather-mcp::get_weather::server.py:1"

    assert shard_for(uid, 4) == shard_for(uid, 4)


def test_shard_for_stays_in_range():
    uids = [f"acme/x::tool_{i}::f.py:{i}" for i in range(200)]

    for num_keys in (1, 2, 3, 7):
        for uid in uids:
            assert 0 <= shard_for(uid, num_keys) < num_keys


def test_shard_for_partitions_are_disjoint_and_exhaustive():
    """Every tool_uid lands in exactly one of the num_keys partitions -- run_parallel_step3.py
    relies on this so the N processes it spawns cover the whole dataset with no overlap and
    no gaps, without needing to persist a partition file anywhere.
    """
    uids = [f"acme/x::tool_{i}::f.py:{i}" for i in range(200)]
    num_keys = 5

    partitions = [{uid for uid in uids if shard_for(uid, num_keys) == i} for i in range(num_keys)]

    assert set().union(*partitions) == set(uids)
    for i in range(num_keys):
        for j in range(i + 1, num_keys):
            assert partitions[i].isdisjoint(partitions[j])


def test_should_skip_true_for_ok_status_regardless_of_retry_failed(tmp_path):
    cp = Checkpoint(tmp_path / "state.json")
    cp.set("k", {"status": "ok"})

    assert should_skip(cp, "k", retry_failed=False) is True
    assert should_skip(cp, "k", retry_failed=True) is True


def test_should_skip_refused_is_never_retried(tmp_path):
    cp = Checkpoint(tmp_path / "state.json")
    cp.set("k", {"status": "refused"})

    assert should_skip(cp, "k", retry_failed=False) is True
    assert should_skip(cp, "k", retry_failed=True) is True


def test_should_skip_error_only_skipped_when_not_retrying(tmp_path):
    cp = Checkpoint(tmp_path / "state.json")
    cp.set("k", {"status": "error"})

    assert should_skip(cp, "k", retry_failed=False) is True
    assert should_skip(cp, "k", retry_failed=True) is False


def test_should_skip_false_when_never_attempted(tmp_path):
    cp = Checkpoint(tmp_path / "state.json")

    assert should_skip(cp, "never-seen", retry_failed=False) is False


def test_run_judge_writes_one_record_per_tool_scenario_and_is_resumable(tmp_path, monkeypatch):
    data_dir = _isolate_data_dirs(tmp_path, monkeypatch)
    rows = [_make_row()]
    checkpoint = Checkpoint(tmp_path / "state.json")
    judge = FakeJudge()

    run_judge(judge, rows, ("description_only", "with_source"), checkpoint, concurrency=2, retry_failed=False)

    assert len(judge.calls) == 2  # one call per scenario
    out_path = data_dir / "evaluations" / "fake-judge.jsonl"
    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(records) == 2
    assert {r["scenario"] for r in records} == {"description_only", "with_source"}
    assert all(r["status"] == "ok" for r in records)

    with_source = next(r for r in records if r["scenario"] == "with_source")
    description_only = next(r for r in records if r["scenario"] == "description_only")
    assert with_source["source_code_sha256"] is not None
    assert description_only["source_code_sha256"] is None

    # Re-running with the same checkpoint must skip everything already completed.
    judge_2 = FakeJudge()
    run_judge(judge_2, rows, ("description_only", "with_source"), checkpoint, concurrency=2, retry_failed=False)
    assert judge_2.calls == []


def test_run_judge_technical_error_is_retried_automatically_next_round(tmp_path, monkeypatch):
    """A technical error (JSON malformed, timeout, etc.) writes no JSONL line and no
    checkpoint entry -- it leaves zero trace, so a plain re-run (no --retry-failed) retries
    it on its own next round. This matters for unattended/parallel runs
    (scripts/run_parallel_step3.py) where nobody is around to pass --retry-failed once
    whatever caused the error clears up. The failure is still visible in
    step3_errors_{judge_id}.jsonl, for a human to notice a pair failing every round.
    """
    data_dir = _isolate_data_dirs(tmp_path, monkeypatch)
    rows = [_make_row()]
    checkpoint = Checkpoint(tmp_path / "state.json")

    run_judge(FakeJudge(fail_on={"get_weather"}), rows, ("description_only",), checkpoint, concurrency=1, retry_failed=False)

    out_path = data_dir / "evaluations" / "fake-judge.jsonl"
    assert not out_path.exists() or out_path.read_text() == ""
    errors_log = tmp_path / "logs" / "step3_errors_fake-judge.jsonl"
    assert errors_log.exists()

    key = checkpoint_key(tool_uid_for(_make_row()), "description_only", "fake-judge")
    assert checkpoint.get(key) is None  # not marked as processed

    # A plain re-run (no --retry-failed) must retry it on its own now.
    recovering = FakeJudge()
    run_judge(recovering, rows, ("description_only",), checkpoint, concurrency=1, retry_failed=False)
    assert len(recovering.calls) == 1
    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert records[-1]["status"] == "ok"


def test_run_judge_retry_failed_flag_still_unsticks_pre_existing_error_checkpoints(tmp_path, monkeypatch):
    """--retry-failed is now only needed to unstick "error" checkpoint entries written by a
    version of this script from before technical errors stopped being checkpointed at all --
    should_skip() must still honor a status="error" entry already sitting in the checkpoint.
    """
    data_dir = _isolate_data_dirs(tmp_path, monkeypatch)
    rows = [_make_row()]
    checkpoint = Checkpoint(tmp_path / "state.json")
    key = checkpoint_key(tool_uid_for(_make_row()), "description_only", "fake-judge")
    checkpoint.set(key, {"status": "error"})

    unretried = FakeJudge()
    run_judge(unretried, rows, ("description_only",), checkpoint, concurrency=1, retry_failed=False)
    assert unretried.calls == []

    recovering = FakeJudge()
    run_judge(recovering, rows, ("description_only",), checkpoint, concurrency=1, retry_failed=True)
    assert len(recovering.calls) == 1
    out_path = data_dir / "evaluations" / "fake-judge.jsonl"
    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert records[-1]["status"] == "ok"


class QuotaExhaustedJudge:
    """Every call fails with the provider's DAILY quota exhausted -- distinct from
    FakeJudge's plain JudgeError, which run_judge must keep dispatching one-per-tool.
    The sleep mirrors real network latency: without it, a single-worker executor could
    race through several queued tasks before run_judge's cancel loop gets a turn.
    """

    def __init__(self):
        self.judge_id = "quota-judge"
        self.provider = "fake"
        self.model_id = "fake-model"
        self.calls: list[str] = []

    def evaluate(self, payload: dict) -> JudgeEvaluation:
        self.calls.append(payload["name"])
        time.sleep(0.02)
        raise JudgeQuotaExhausted("gemini daily quota exhausted: simulated")


def test_run_judge_stops_batch_early_on_daily_quota_exhaustion(tmp_path, monkeypatch):
    """A JudgeQuotaExhausted mid-batch must not crash run_judge (unlike a plain JudgeError,
    it signals every remaining call would fail identically) -- it should cancel whatever
    is still queued instead of dispatching the rest of a --limit batch just to collect
    more copies of the same 429. concurrency=1 makes the guarantee deterministic: only the
    task already running when the quota error lands can possibly get an extra one racing
    in behind it, and the other three of five are still queued and so definitely
    cancellable within a single worker.
    """
    data_dir = _isolate_data_dirs(tmp_path, monkeypatch)
    rows = [_make_row(name=f"tool_{i}") for i in range(5)]
    checkpoint = Checkpoint(tmp_path / "state.json")
    judge = QuotaExhaustedJudge()

    run_judge(judge, rows, ("description_only",), checkpoint, concurrency=1, retry_failed=False)

    assert len(judge.calls) <= 2  # at most the triggering call plus one already-dequeued race
    assert len(judge.calls) < len(rows)  # the rest were cancelled, not just slow

    # Neither the cancelled tasks nor the ones that were already in flight when the quota
    # error landed leave any trace -- no JSONL line, no checkpoint entry -- so a plain
    # re-run, no --retry-failed, retries every one of them once the quota resets.
    out_path = data_dir / "evaluations" / "quota-judge.jsonl"
    assert not out_path.exists() or out_path.read_text() == ""

    for i in range(len(rows)):
        key = checkpoint_key(tool_uid_for(_make_row(name=f"tool_{i}")), "description_only", "quota-judge")
        assert checkpoint.get(key) is None
