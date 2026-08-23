from __future__ import annotations

import argparse
import sys

from mcp_pipeline.collection.checkpoint import Checkpoint
from mcp_pipeline.collection.dedupe_rank import dedupe_and_rank, write_jsonl
from mcp_pipeline.config import (
    DATA_DIR,
    STATE_DIR,
    Signals,
    ensure_dirs,
    get_github_token,
)
from mcp_pipeline.github.graphql_client import GraphQLClient
from mcp_pipeline.github.queries import SMOKE_TEST_QUERY
from mcp_pipeline.github.search_manifest import search_by_manifest_signals
from mcp_pipeline.github.search_text import search_by_text_signals
from mcp_pipeline.github.search_topics import search_by_topics
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("step1")


def smoke_test(client: GraphQLClient) -> None:
    data = client.execute(SMOKE_TEST_QUERY)
    rate_limit = data["rateLimit"]
    logger.info(
        "Autenticado como %s | rate limit: %s/%s restantes (reset em %s)",
        data["viewer"]["login"],
        rate_limit["remaining"],
        rate_limit["limit"],
        rate_limit["resetAt"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 1: mineração de repositórios MCP via GraphQL")
    parser.add_argument(
        "--smoke-test-only",
        action="store_true",
        help="Só valida autenticação e rate limit, sem rodar a busca completa.",
    )
    args = parser.parse_args()

    ensure_dirs()
    signals = Signals.load()
    token = get_github_token()
    client = GraphQLClient(token)

    logger.info("Rodando teste de fumaça...")
    smoke_test(client)
    if args.smoke_test_only:
        return

    checkpoint = Checkpoint(STATE_DIR / "step1_progress.json")

    logger.info(
        "Buscando repositórios: %s tópicos, %s sinais textuais, %s sinais de manifesto, "
        "linguagens=%s, min_stars=%s",
        len(signals.topics),
        len(signals.text_signals),
        len(signals.manifest_signals),
        signals.target_languages,
        signals.min_stars,
    )

    all_candidates = (
        list(search_by_topics(client, checkpoint, signals))
        + list(search_by_text_signals(client, checkpoint, signals))
        + list(search_by_manifest_signals(client, checkpoint, signals, token))
    )
    logger.info("Total de resultados brutos (com duplicatas entre sub-queries): %s", len(all_candidates))

    pool, selected = dedupe_and_rank(all_candidates, signals)

    write_jsonl(pool, DATA_DIR / "candidate_pool.jsonl")
    write_jsonl(selected, DATA_DIR / "selected_repos.jsonl")

    logger.info(
        "Etapa 1 concluída: %s repos no pool completo, %s selecionados (top %s) -> %s",
        len(pool),
        len(selected),
        signals.top_n,
        DATA_DIR / "selected_repos.jsonl",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Etapa 1 falhou")
        sys.exit(1)
