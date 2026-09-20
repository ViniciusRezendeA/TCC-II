from __future__ import annotations

from unittest.mock import MagicMock

from mcp_pipeline.evaluation.judges.base import RubricScores
from mcp_pipeline.evaluation.judges.ollama_judge import OllamaJudge


def test_ollama_judge_maps_wrapper_response_to_evaluation(monkeypatch):
    component = {"score": 4, "reasoning": "Adequate."}
    scores = {
        "purpose": component,
        "guidelines": component,
        "limitations": component,
        "parameter_explanation": component,
        "length_completeness": component,
        "examples": component,
    }
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "model": "qwen3:14b",
        "response": __import__("json").dumps(scores),
        "prompt_tokens": 12,
        "completion_tokens": 34,
    }
    post = MagicMock(return_value=fake_response)
    monkeypatch.setattr("mcp_pipeline.evaluation.judges.ollama_judge.requests.post", post)

    result = OllamaJudge("qwen3-14b-ollama", "qwen3:14b").evaluate(
        {"name": "get_weather", "server_name": "acme/weather", "description": "Fetch weather."}
    )

    assert result.scores.purpose.score == 4
    assert result.input_tokens == 12
    assert result.output_tokens == 34
    assert result.raw_model_version == "qwen3:14b"
    request = post.call_args.kwargs["json"]
    assert request["model"] == "qwen3:14b"
    assert request["format"] == RubricScores.model_json_schema()