from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from mcp_pipeline.clone.clone_manager import META_FILENAME, RepoMeta
from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.logging_setup import setup_logging
from mcp_pipeline.pipeline.run_step2 import TOOLS_FILENAME

logger = setup_logging("assemble_dataset")

SCHEMA_VERSION = "1.0"
EXTRACTOR_VERSION = "0.1.0"


def assemble_dataset(repos_root: Path, output_path: Path) -> int:
    """Concatenates every repo's tools.jsonl (Etapa 2 output) into one final
    dataset.jsonl, denormalizing repo metadata into each row so Etapa 3
    never needs to join back to repo_meta.json. Repos not yet processed by
    Etapa 2 (no tools.jsonl) are silently skipped — this can be re-run at
    any point during a partial Etapa 2 run to get a snapshot of progress.
    """
    count = 0
    extracted_at = datetime.datetime.now(datetime.UTC).isoformat()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for meta_file in sorted(repos_root.glob(f"*/{META_FILENAME}")):
            tools_file = meta_file.parent / TOOLS_FILENAME
            if not tools_file.exists():
                continue

            meta = RepoMeta.from_meta_file(meta_file)
            for line in tools_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
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

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatena a saída da Etapa 2 em data/dataset.jsonl")
    parser.parse_args()

    repos_root = DATA_DIR / "repos"
    output_path = DATA_DIR / "dataset.jsonl"
    count = assemble_dataset(repos_root, output_path)
    logger.info("dataset.jsonl montado: %s registros (1 por tool) -> %s", count, output_path)


if __name__ == "__main__":
    main()
