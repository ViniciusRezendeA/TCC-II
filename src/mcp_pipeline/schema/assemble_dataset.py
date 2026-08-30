from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from mcp_pipeline.clone.clone_manager import META_FILENAME, RepoMeta
from mcp_pipeline.config import DATA_DIR, Signals
from mcp_pipeline.logging_setup import setup_logging
from mcp_pipeline.pipeline.run_step2 import TOOLS_FILENAME

logger = setup_logging("assemble_dataset")

SCHEMA_VERSION = "1.0"
EXTRACTOR_VERSION = "0.1.0"


def assemble_dataset(repos_root: Path, output_path: Path, min_tools: int = 1) -> int:
    """Concatenates every repo's tools.jsonl (Etapa 2 output) into one final
    dataset.jsonl, denormalizing repo metadata into each row so Etapa 3
    never needs to join back to repo_meta.json. Repos not yet processed by
    Etapa 2 (no tools.jsonl) are silently skipped — this can be re-run at
    any point during a partial Etapa 2 run to get a snapshot of progress.

    Repos processed by Etapa 2 but whose tool count falls below `min_tools`
    (signals.min_tools, e.g. the "candidate wasn't actually an MCP server"
    case — see tool_detector.py) are excluded too, and counted separately in
    the returned/logged stats so they aren't confused with not-yet-processed
    repos.
    """
    count = 0
    excluded_below_min_tools = 0
    extracted_at = datetime.datetime.now(datetime.UTC).isoformat()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for meta_file in sorted(repos_root.glob(f"*/{META_FILENAME}")):
            tools_file = meta_file.parent / TOOLS_FILENAME
            if not tools_file.exists():
                continue

            lines = [line for line in tools_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) < min_tools:
                excluded_below_min_tools += 1
                continue

            meta = RepoMeta.from_meta_file(meta_file)
            for line in lines:
                record = json.loads(line)
                combined = {
                    "schema_version": SCHEMA_VERSION,
                    "repo": meta.to_dict(),
                    "tool": record["tool"],
                    "call_graph": record["call_graph"],
                    "extraction_meta": {"extracted_at": extracted_at, "extractor_version": EXTRACTOR_VERSION},
                }
                out.write(json.dumps(combined, ensure_ascii=False) + "\n")
                count += 1

    if excluded_below_min_tools:
        logger.info(
            "%s repo(s) processados pela Etapa 2 excluídos do dataset por terem menos de "
            "min_tools=%s tool(s)",
            excluded_below_min_tools,
            min_tools,
        )

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatena a saída da Etapa 2 em data/dataset.jsonl")
    parser.parse_args()

    signals = Signals.load()
    repos_root = DATA_DIR / "repos"
    output_path = DATA_DIR / "dataset.jsonl"
    count = assemble_dataset(repos_root, output_path, min_tools=signals.min_tools)
    logger.info("dataset.jsonl montado: %s registros (1 por tool) -> %s", count, output_path)


if __name__ == "__main__":
    main()
