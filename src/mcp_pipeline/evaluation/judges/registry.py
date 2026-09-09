from __future__ import annotations

from pathlib import Path

import yaml

from mcp_pipeline.config import CONFIG_DIR
from mcp_pipeline.evaluation.judges.base import Judge
from mcp_pipeline.evaluation.judges.gemini_judge import GeminiJudge
from mcp_pipeline.evaluation.judges.llama_judge import LlamaJudge
from mcp_pipeline.evaluation.judges.qwen_judge import QwenJudge

# Same "registry dict keyed by a short label" idiom as extraction/tool_detector.py's
# LANGUAGE_ADAPTERS -- adding a provider é um novo *_judge.py class + uma entrada aqui +
# um bloco em judges.yaml; nada mais muda neste arquivo ou em run_step3.py.
PROVIDER_CLASSES: dict[str, type] = {
    "google": GeminiJudge,
    "llama": LlamaJudge,
    "qwen": QwenJudge,
}

# Any judges.yaml key beyond these is passed straight through as a kwarg to the provider's
# constructor (e.g. `requests_per_minute` for GeminiJudge) -- keeps per-model tuning in the
# config file instead of code, same as model_id.
_ENTRY_KEYS = {"id", "provider", "model_id", "enabled"}


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
        extra_kwargs = {k: v for k, v in entry.items() if k not in _ENTRY_KEYS}
        judges.append(judge_cls(judge_id=entry["id"], model_id=entry["model_id"], **extra_kwargs))

    return judges
