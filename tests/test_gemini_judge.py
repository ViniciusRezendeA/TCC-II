from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcp_pipeline.evaluation.judges.base import JudgeError, JudgeRefusal, RubricScores
from mcp_pipeline.evaluation.judges.gemini_judge import GeminiJudge, _RateLimiter


def _make_judge(monkeypatch, **kwargs) -> GeminiJudge:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-tests")
    return GeminiJudge(judge_id="gemini-2.5-flash-lite", model_id="gemini-2.5-flash-lite", **kwargs)


def _rubric_scores() -> RubricScores:
    component = {"score": 5, "reasoning": "Excellent."}
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


def test_rubric_schema_uses_integer_score_range_for_gemini():
    score_schema = RubricScores.model_json_schema()["$defs"]["ComponentScore"]["properties"]["score"]

    assert score_schema == {
        "maximum": 5,
        "minimum": 1,
        "title": "Score",
        "type": "integer",
    }


def test_evaluate_returns_scores_and_usage_on_success(monkeypatch):
    judge = _make_judge(monkeypatch)
    fake_response = SimpleNamespace(
        parsed=_rubric_scores(),
        candidates=[SimpleNamespace(finish_reason="STOP")],
        model_version="gemini-2.5-flash-lite-001",
        usage_metadata=SimpleNamespace(prompt_token_count=120, candidates_token_count=180, cached_content_token_count=0),
    )
    monkeypatch.setattr(judge._client.models, "generate_content", MagicMock(return_value=fake_response))

    result = judge.evaluate({"name": "get_weather", "server_name": "acme/weather-mcp", "description": "..."})

    assert result.scores.purpose.score == 5
    assert result.input_tokens == 120
    assert result.output_tokens == 180
    assert result.raw_model_version == "gemini-2.5-flash-lite-001"


def test_evaluate_raises_judge_refusal_on_safety_finish_reason(monkeypatch):
    judge = _make_judge(monkeypatch)
    fake_response = SimpleNamespace(
        parsed=None,
        candidates=[SimpleNamespace(finish_reason="PROHIBITED_CONTENT")],
        model_version="gemini-2.5-flash-lite-001",
        usage_metadata=SimpleNamespace(prompt_token_count=120, candidates_token_count=0, cached_content_token_count=0),
    )
    monkeypatch.setattr(judge._client.models, "generate_content", MagicMock(return_value=fake_response))

    with pytest.raises(JudgeRefusal) as exc_info:
        judge.evaluate({"name": "nmap_scan", "server_name": "acme/pentest-mcp", "description": "..."})

    assert exc_info.value.category == "PROHIBITED_CONTENT"


def test_evaluate_raises_judge_error_on_non_safety_empty_parse(monkeypatch):
    judge = _make_judge(monkeypatch)
    fake_response = SimpleNamespace(
        parsed=None,
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
        model_version="gemini-2.5-flash-lite-001",
        usage_metadata=SimpleNamespace(prompt_token_count=120, candidates_token_count=999, cached_content_token_count=0),
    )
    monkeypatch.setattr(judge._client.models, "generate_content", MagicMock(return_value=fake_response))

    with pytest.raises(JudgeError):
        judge.evaluate({"name": "get_weather", "server_name": "acme/weather-mcp", "description": "..."})


def test_rate_limiter_paces_calls_evenly_regardless_of_wall_clock(monkeypatch):
    """--concurrency alone can burst well past a model's RPM if responses come back fast
    (observed live: 429s on gemini-3.5-flash-lite even at --concurrency 3). _RateLimiter
    must space calls by 60/N seconds apart, not just let N through per wall-clock minute.
    """
    fake_now = 1000.0
    monkeypatch.setattr("mcp_pipeline.evaluation.judges.gemini_judge.time.monotonic", lambda: fake_now)
    sleeps: list[float] = []
    monkeypatch.setattr("mcp_pipeline.evaluation.judges.gemini_judge.time.sleep", sleeps.append)

    limiter = _RateLimiter(requests_per_minute=60)  # 1 call/second
    limiter.wait()  # first call: clock is free, no wait
    limiter.wait()  # second call: must wait for the 1s slot after the first
    limiter.wait()  # third call: must wait for the 2s slot, cumulative

    assert sleeps == [1.0, 2.0]


def test_evaluate_waits_on_rate_limiter_before_calling_the_api(monkeypatch):
    judge = _make_judge(monkeypatch, requests_per_minute=30)
    wait_mock = MagicMock()
    monkeypatch.setattr(judge._rate_limiter, "wait", wait_mock)
    fake_response = SimpleNamespace(
        parsed=_rubric_scores(),
        candidates=[SimpleNamespace(finish_reason="STOP")],
        model_version="gemini-2.5-flash-lite-001",
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1, cached_content_token_count=0),
    )
    monkeypatch.setattr(judge._client.models, "generate_content", MagicMock(return_value=fake_response))

    judge.evaluate({"name": "get_weather", "server_name": "acme/weather-mcp", "description": "..."})

    wait_mock.assert_called_once()
