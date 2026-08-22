from __future__ import annotations

import json

from mcp_pipeline.collection.checkpoint import Checkpoint
from mcp_pipeline.evaluation.judges.base import (
    JudgeError,
    JudgeEvaluation,
    RubricScores,
)
from mcp_pipeline.pipeline.run_step3 import (
    checkpoint_key,
    run_judge,
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


def test_checkpoint_key_embeds_prompt_version():
    from mcp_pipeline.evaluation.prompts import PROMPT_VERSION

    key = checkpoint_key("acme/x::t::f.py:1", "description_only", "claude-haiku-4-5")

    assert key.endswith(f"::{PROMPT_VERSION}")


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


def test_run_judge_error_is_retried_only_with_retry_failed_flag(tmp_path, monkeypatch):
    data_dir = _isolate_data_dirs(tmp_path, monkeypatch)
    rows = [_make_row()]
    checkpoint = Checkpoint(tmp_path / "state.json")

    run_judge(FakeJudge(fail_on={"get_weather"}), rows, ("description_only",), checkpoint, concurrency=1, retry_failed=False)

    out_path = data_dir / "evaluations" / "fake-judge.jsonl"
    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert records[-1]["status"] == "error"
    errors_log = tmp_path / "logs" / "step3_errors_fake-judge.jsonl"
    assert errors_log.exists()

    # A plain re-run (no --retry-failed) must not retry a logged technical error.
    unretried = FakeJudge(fail_on={"get_weather"})
    run_judge(unretried, rows, ("description_only",), checkpoint, concurrency=1, retry_failed=False)
    assert unretried.calls == []

    # --retry-failed must retry it -- and here it succeeds.
    recovering = FakeJudge()
    run_judge(recovering, rows, ("description_only",), checkpoint, concurrency=1, retry_failed=True)
    assert len(recovering.calls) == 1
    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert records[-1]["status"] == "ok"
