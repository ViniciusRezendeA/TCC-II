from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
STATE_DIR = PROJECT_ROOT / "state"
LOGS_DIR = PROJECT_ROOT / "logs"

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

# Loaded once at import time (not lazily inside get_github_token(), which only Etapa 1
# called) so every entrypoint gets .env populated -- Etapa 3's judges read their own API
# key env vars directly via each provider's SDK, with no getter in this module to trigger
# a lazy load.
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class ManifestSignal:
    """One text_signals entry paired with the library-manifest file(s) it's
    expected to appear in for its ecosystem (e.g. "@modelcontextprotocol/sdk"
    in package.json), used to drive the REST code-search fallback in
    github/search_manifest.py.
    """

    signal: str
    file_qualifiers: list[str]
    languages: list[str]


@dataclass(frozen=True)
class Signals:
    topics: list[str]
    text_signals: list[str]
    manifest_signals: list[ManifestSignal]
    target_languages: list[str]
    min_stars: int
    top_n: int
    result_count_warning_threshold: int


    @classmethod
    def load(cls, path: Path | None = None) -> Signals:
        path = path or (CONFIG_DIR / "mcp_signals.yaml")

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        manifest_signals = [
            ManifestSignal(
                signal=entry["signal"],
                file_qualifiers=entry["file_qualifiers"],
                languages=entry["languages"],
            )
            for entry in raw["manifest_signals_v2"]
        ]

        return cls(
            topics=raw["topics"],
            text_signals=raw["text_signals"],
            manifest_signals=manifest_signals,
            target_languages=raw["target_languages"],
            min_stars=raw["min_stars"],
            top_n=raw["top_n"],
            result_count_warning_threshold=raw["result_count_warning_threshold"],
        )




def get_github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN não encontrado. Copie .env.example para .env e "
            "preencha com um Personal Access Token do GitHub (somente leitura, "
            "escopo de repositórios públicos)."
        )
    return token


def ensure_dirs() -> None:
    for d in (DATA_DIR, DATA_DIR / "raw" / "search_pages", DATA_DIR / "repos", STATE_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
