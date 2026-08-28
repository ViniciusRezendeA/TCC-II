from __future__ import annotations

import argparse
import sys

from mcp_pipeline.collection.checkpoint import Checkpoint
from mcp_pipeline.collection.dedupe_rank import dedupe_and_rank, read_jsonl, write_jsonl
from mcp_pipeline.config import (
    DATA_DIR,
    STATE_DIR,
    Signals,
    ensure_dirs,
    get_github_token,
)
from mcp_pipeline.github.graphql_client import GraphQLClient
from mcp_pipeline.github.models import RepoCandidate
from mcp_pipeline.github.queries import SMOKE_TEST_QUERY
from mcp_pipeline.github.search_manifest import (
    build_candidates_from_cached_pages,
    search_by_manifest_signals,
)
from mcp_pipeline.github.search_text import search_by_text_signals
from mcp_pipeline.github.search_topics import search_by_topics
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("step1")

# Every candidate as hydrated by the last full (non-cached) run, before
# dedupe/filter — lets --use-cache re-run just the selection step (dedupe +
# filter_and_rank) against already-paid-for GraphQL/REST calls, e.g. while
# iterating on min_stars/top_n in mcp_signals.yaml or on filter_and_rank
# itself, without re-spending rate limit. Combines all 3 sources (topics,
# text_signals, manifest_signals) — renamed from the earlier
# manifest_candidates.jsonl once search_by_topics/search_by_text_signals
# were wired back in alongside the REST manifest search below, since the
# cache is no longer manifest-only.
RAW_CANDIDATES_PATH = DATA_DIR / "raw" / "all_candidates.jsonl"


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
    mutex = parser.add_mutually_exclusive_group()
    mutex.add_argument(
        "--use-cache",
        action="store_true",
        help=(
            f"Pula a busca (REST + hidratação GraphQL) e reusa os candidatos já "
            f"hidratados salvos em {RAW_CANDIDATES_PATH} pela última rodada sem "
            f"--use-cache/--offline, rodando só a seleção (dedupe + filtro). Útil para "
            f"iterar em min_stars/top_n ou em filter_and_rank sem gastar rate limit de novo."
        ),
    )
    mutex.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Monta candidate_pool.jsonl/selected_repos.jsonl só a partir dos search_pages "
            "REST já em cache — zero chamadas de rede (nem REST nem GraphQL). Só filtra "
            "por fork (único sinal de qualidade disponível sem hidratar); NÃO tem "
            "stargazer_count/primary_language reais (ficam com valor neutro: 0/None) e "
            "não aplica min_stars. Rankeia por número de signals que bateram em cada repo, "
            "não por estrelas. Preview rápido só — para dados reais rode sem --offline."
        ),
    )
    args = parser.parse_args()

    ensure_dirs()
    signals = Signals.load()

    if args.offline:
        pool = build_candidates_from_cached_pages(signals)
        selected = pool[: signals.top_n]
        write_jsonl(pool, DATA_DIR / "candidate_pool.jsonl")
        write_jsonl(selected, DATA_DIR / "selected_repos.jsonl")
        logger.info(
            "Etapa 1 (--offline, só REST, sem estrelas/linguagem reais) concluída: "
            "%s repos no pool, %s selecionados (top %s) -> %s",
            len(pool),
            len(selected),
            signals.top_n,
            DATA_DIR / "selected_repos.jsonl",
        )
        return

    if args.use_cache:
        if not RAW_CANDIDATES_PATH.exists():
            logger.error(
                "--use-cache passado mas %s não existe — rode sem --use-cache pelo menos "
                "uma vez antes para gerar o cache.",
                RAW_CANDIDATES_PATH,
            )
            sys.exit(1)
        all_candidates = read_jsonl(RAW_CANDIDATES_PATH)
        logger.info(
            "Usando %s candidatos em cache de %s (--use-cache, sem busca)",
            len(all_candidates),
            RAW_CANDIDATES_PATH,
        )
    else:
        token = get_github_token()
        client = GraphQLClient(token)

        logger.info("Rodando teste de fumaça...")
        smoke_test(client)
        if args.smoke_test_only:
            return

        checkpoint = Checkpoint(STATE_DIR / "step1_progress.json")

        logger.info(
            "Buscando repositórios: %s sinais de manifesto, linguagens=%s, min_stars=%s",
            len(signals.manifest_signals),
            signals.target_languages,
            signals.min_stars,
        )

        all_candidates = list(search_by_manifest_signals(client, checkpoint, signals, token))
        write_jsonl(all_candidates, RAW_CANDIDATES_PATH)
        logger.info("Total de resultados brutos (com duplicatas entre sub-queries): %s", len(all_candidates))

    _log_candidates_per_source(all_candidates)

    pool, selected = dedupe_and_rank(all_candidates, signals)

    write_jsonl(pool, DATA_DIR / "candidate_pool.jsonl")
    write_jsonl(selected, DATA_DIR / "selected_repos.jsonl")

    logger.info(
        "Etapa 1 concluída: %s repos no pool completo (passaram filtro de fork/estrelas/linguagem) -> "
        "%s selecionados (top %s) serão clonados e terão tools extraídas na Etapa 2 -> %s",
        len(pool),
        len(selected),
        signals.top_n,
        DATA_DIR / "selected_repos.jsonl",
    )


def _log_candidates_per_source(all_candidates: list[RepoCandidate]) -> None:
    """Breaks down raw candidate counts by filter source (manifest/topic/text),
    derived from each candidate's matched_signals prefix rather than hardcoded
    to whichever sources actually ran, so this stays correct if topics/text_signals
    get wired back into main() alongside manifest_signals.
    """
    per_source: dict[str, int] = {}
    for candidate in all_candidates:
        for signal in candidate.matched_signals:
            source = signal.split(":", 1)[0]
            per_source[source] = per_source.get(source, 0) + 1
    logger.info("Candidatos brutos por fonte de filtro: %s", per_source)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Etapa 1 falhou")
        sys.exit(1)
