from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcp_pipeline.evaluation.judges.anthropic_judge import AnthropicJudge
from mcp_pipeline.evaluation.judges.base import JudgeRefusal, RubricScores


def _make_judge(monkeypatch) -> AnthropicJudge:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    return AnthropicJudge(judge_id="claude-haiku-4-5", model_id="claude-haiku-4-5")


def _rubric_scores() -> RubricScores:
    component = {"score": 4, "reasoning": "Clear enough."}
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


def test_evaluate_returns_scores_and_usage_on_success(monkeypatch):
    judge = _make_judge(monkeypatch)
    fake_response = SimpleNamespace(
        stop_reason="end_turn",
        stop_details=None,
        parsed_output=_rubric_scores(),
        model="claude-haiku-4-5-20260101",
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=200, cache_read_input_tokens=50, cache_creation_input_tokens=0
        ),
    )
    monkeypatch.setattr(judge._client.messages, "parse", MagicMock(return_value=fake_response))

    result = judge.evaluate({"name": "get_weather", "server_name": "acme/weather-mcp", "description": "..."})

    assert result.scores.purpose.score == 4
    assert result.input_tokens == 100
    assert result.output_tokens == 200
    assert result.cache_read_tokens == 50
    assert result.raw_model_version == "claude-haiku-4-5-20260101"


def test_evaluate_raises_judge_refusal_on_refusal_stop_reason(monkeypatch):
    judge = _make_judge(monkeypatch)
    fake_response = SimpleNamespace(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber"),
        parsed_output=None,
        model="claude-haiku-4-5-20260101",
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
        ),
    )
    monkeypatch.setattr(judge._client.messages, "parse", MagicMock(return_value=fake_response))

    with pytest.raises(JudgeRefusal) as exc_info:
        judge.evaluate({"name": "nmap_scan", "server_name": "acme/pentest-mcp", "description": "..."})

    assert exc_info.value.category == "cyber"


def test_evaluate_sends_rubric_as_cached_system_block_and_payload_as_user_turn(monkeypatch):
    judge = _make_judge(monkeypatch)
    fake_response = SimpleNamespace(
        stop_reason="end_turn",
        stop_details=None,
        parsed_output=_rubric_scores(),
        model="claude-haiku-4-5-20260101",
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=200, cache_read_input_tokens=0, cache_creation_input_tokens=1200
        ),
    )
    mock_parse = MagicMock(return_value=fake_response)
    monkeypatch.setattr(judge._client.messages, "parse", mock_parse)

    judge.evaluate({"name": "get_weather", "server_name": "acme/weather-mcp", "description": "..."})

    _, kwargs = mock_parse.call_args
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert '"name": "get_weather"' in kwargs["messages"][0]["content"]
