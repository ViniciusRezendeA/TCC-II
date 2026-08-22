from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcp_pipeline.evaluation.judges.base import JudgeRefusal, RubricScores
from mcp_pipeline.evaluation.judges.openai_judge import OpenAIJudge


def _make_judge(monkeypatch) -> OpenAIJudge:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-tests")
    return OpenAIJudge(judge_id="gpt-4.1-mini", model_id="gpt-4.1-mini")


def _rubric_scores() -> RubricScores:
    component = {"score": 3, "reasoning": "Adequate."}
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


def _usage(cached_tokens: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=150,
        output_tokens=250,
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens, cache_write_tokens=0),
    )


def test_evaluate_returns_scores_and_usage_on_success(monkeypatch):
    judge = _make_judge(monkeypatch)
    fake_response = SimpleNamespace(
        output_parsed=_rubric_scores(),
        output=[],
        status="completed",
        model="gpt-4.1-mini-2026-01-01",
        usage=_usage(cached_tokens=64),
    )
    monkeypatch.setattr(judge._client.responses, "parse", MagicMock(return_value=fake_response))

    result = judge.evaluate({"name": "get_weather", "server_name": "acme/weather-mcp", "description": "..."})

    assert result.scores.purpose.score == 3
    assert result.input_tokens == 150
    assert result.output_tokens == 250
    assert result.cache_read_tokens == 64
    assert result.raw_model_version == "gpt-4.1-mini-2026-01-01"


def test_evaluate_raises_judge_refusal_when_output_has_refusal_block(monkeypatch):
    judge = _make_judge(monkeypatch)
    refusal_block = SimpleNamespace(type="refusal", refusal="cannot help with this request")
    message_item = SimpleNamespace(content=[refusal_block])
    fake_response = SimpleNamespace(
        output_parsed=None,
        output=[message_item],
        status="completed",
        model="gpt-4.1-mini-2026-01-01",
        usage=_usage(),
    )
    monkeypatch.setattr(judge._client.responses, "parse", MagicMock(return_value=fake_response))

    with pytest.raises(JudgeRefusal) as exc_info:
        judge.evaluate({"name": "nmap_scan", "server_name": "acme/pentest-mcp", "description": "..."})

    assert "cannot help" in exc_info.value.category


def test_evaluate_raises_judge_error_when_no_parsed_output_and_no_refusal_found(monkeypatch):
    judge = _make_judge(monkeypatch)
    fake_response = SimpleNamespace(
        output_parsed=None, output=[], status="incomplete", model="gpt-4.1-mini-2026-01-01", usage=_usage()
    )
    monkeypatch.setattr(judge._client.responses, "parse", MagicMock(return_value=fake_response))

    from mcp_pipeline.evaluation.judges.base import JudgeError

    with pytest.raises(JudgeError):
        judge.evaluate({"name": "get_weather", "server_name": "acme/weather-mcp", "description": "..."})
