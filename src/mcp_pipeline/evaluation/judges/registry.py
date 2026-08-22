from __future__ import annotations

from pathlib import Path

import yaml

from mcp_pipeline.config import CONFIG_DIR
from mcp_pipeline.evaluation.judges.anthropic_judge import AnthropicJudge
from mcp_pipeline.evaluation.judges.base import Judge
from mcp_pipeline.evaluation.judges.gemini_judge import GeminiJudge
from mcp_pipeline.evaluation.judges.openai_judge import OpenAIJudge

# Same "registry dict keyed by a short label" idiom as extraction/tool_detector.py's
# LANGUAGE_ADAPTERS -- adding a 4th provider is one new *_judge.py class + one entry here +
# one block in judges.yaml; nothing else in this file or in run_step3.py changes.
PROVIDER_CLASSES: dict[str, type] = {
    "anthropic": AnthropicJudge,
    "openai": OpenAIJudge,
    "google": GeminiJudge,
}


def load_judges(config_path: Path | None = None, only: set[str] | None = None) -> list[Judge]:
    """Reads config/judges.yaml, instantiates every entry with enabled: true (or, if `only`
    is given, every entry whose id is in `only`, regardless of its enabled flag -- lets
    `run_step3.py --judges <id>` target a single judge for a pilot run without editing the
    config file).
    """
    config_path = config_path or (CONFIG_DIR / "judges.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    judges: list[Judge] = []
    for entry in raw["judges"]:
        if only is not None:
            if entry["id"] not in only:
                continue
        elif not entry.get("enabled", False):
            continue

        provider = entry["provider"]
        if provider not in PROVIDER_CLASSES:
            raise ValueError(f"provedor desconhecido {provider!r} para o juiz {entry['id']!r}")

        judge_cls = PROVIDER_CLASSES[provider]
        judges.append(judge_cls(judge_id=entry["id"], model_id=entry["model_id"]))

    return judges
