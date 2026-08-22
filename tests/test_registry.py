from __future__ import annotations

from pathlib import Path

import pytest

from mcp_pipeline.evaluation.judges.registry import load_judges


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "judges.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_judges_only_instantiates_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    config = _write_config(
        tmp_path,
        """
judges:
  - id: claude-haiku-4-5
    provider: anthropic
    model_id: claude-haiku-4-5
    enabled: true
  - id: gpt-4.1-mini
    provider: openai
    model_id: gpt-4.1-mini
    enabled: false
""",
    )

    judges = load_judges(config_path=config)

    assert [j.judge_id for j in judges] == ["claude-haiku-4-5"]


def test_only_filter_overrides_enabled_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    config = _write_config(
        tmp_path,
        """
judges:
  - id: claude-haiku-4-5
    provider: anthropic
    model_id: claude-haiku-4-5
    enabled: true
  - id: gpt-4.1-mini
    provider: openai
    model_id: gpt-4.1-mini
    enabled: false
""",
    )

    judges = load_judges(config_path=config, only={"gpt-4.1-mini"})

    assert [j.judge_id for j in judges] == ["gpt-4.1-mini"]


def test_load_judges_raises_on_unknown_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    config = _write_config(
        tmp_path,
        """
judges:
  - id: mystery
    provider: mystery_provider
    model_id: mystery-1
    enabled: true
""",
    )

    with pytest.raises(ValueError, match="mystery_provider"):
        load_judges(config_path=config)


def test_load_judges_sets_provider_and_model_id_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    config = _write_config(
        tmp_path,
        """
judges:
  - id: gemini-2.5-flash-lite
    provider: google
    model_id: gemini-2.5-flash-lite
    enabled: true
""",
    )

    (judge,) = load_judges(config_path=config)

    assert judge.provider == "google"
    assert judge.model_id == "gemini-2.5-flash-lite"
