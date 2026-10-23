#!/usr/bin/env python
"""Seleciona as 100 ferramentas da amostra de consenso (data/analysis/consenso_amostra.jsonl)
para a verificação manual da classificação dirigida, conforme o n=100 determinado pela
Fórmula de Cochran já documentada na metodologia do TCC (Seção 4.5).

Critério de seleção: amostragem estratificada por categoria de mecanismo (as 16 categorias de
data/analysis/consenso_categorias_reconciliadas.json, mais "none" como décimo sétimo estrato),
com vagas distribuídas o mais igualmente possível entre os estratos -- não proporcionalmente à
frequência de cada categoria -- para que as 100 ferramentas não acabem representando
desproporcionalmente uma única categoria, por ser também a mais comum na população. Dentro de
cada estrato, a seleção é gulosa e prioriza, nesta ordem: (1) ferramentas de repositórios ainda
não usados em nenhum outro ponto da amostra de 100, para não concentrar em poucos repositórios;
(2) ferramentas cujo(s) componente(s) divergente(s) da rubrica de Hasan et al. (2026) (Purpose,
Guidelines, Limitations, Parameter Explanation, Length & Completeness, Examples) ainda não
apareceram muito dentro daquele mesmo estrato, para não deixar a amostra de uma categoria vir
sempre do mesmo componente.

Com --manter, as ferramentas de uma amostra anterior (mesmo formato da saída) que ainda existem
na amostra de consenso, com a mesma categoria, entram primeiro e já contam para as vagas do seu
estrato, para os repositórios e para os componentes usados; só as vagas restantes são sorteadas.
Serve para recompor a amostra quando a amostra de consenso muda (ex.: um juiz concluído depois)
sem descartar verificações já feitas que continuam válidas.

Uso:
  uv run python -m scripts.select_consenso_verification_sample
  uv run python -m scripts.select_consenso_verification_sample --n 100 --seed 42
  uv run python -m scripts.select_consenso_verification_sample --manter manter.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("select_consenso_verification_sample")

N_AMOSTRA = 100
SEED = 42


def _repo_from_tool_uid(tool_uid: str) -> str:
    return tool_uid.split("::", 1)[0]


def _componentes_divergentes(avaliacoes_brutas: list[dict]) -> set[str]:
    comps: set[str] = set()
    for av in avaliacoes_brutas:
        comps.update(av["scores"].keys())
    return comps


def _alocar_por_estrato(tamanhos: dict[str, int], n_total: int) -> dict[str, int]:
    """Distribui n_total vagas entre os estratos o mais igualmente possível, com o restante
    indo primeiro aos maiores estratos (que têm mais candidatos sobrando para absorver uma
    vaga extra sem esgotar a diversidade interna). Nunca aloca mais vagas do que ferramentas
    existem no estrato.
    """
    nomes = list(tamanhos.keys())
    base = n_total // len(nomes)
    resto = n_total - base * len(nomes)
    alocacao = {nome: min(base, tamanhos[nome]) for nome in nomes}
    for nome in sorted(nomes, key=lambda n: -tamanhos[n]):
        if resto <= 0:
            break
        if alocacao[nome] < tamanhos[nome]:
            alocacao[nome] += 1
            resto -= 1
    return alocacao


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seleciona amostra estratificada (categoria, componente da rubrica, repositório) para verificação manual da classificação dirigida."
    )
    parser.add_argument("--amostra", type=Path, default=None, help="default: data/analysis/consenso_amostra.jsonl")
    parser.add_argument("--justificativas", type=Path, default=None, help="default: data/analysis/consenso_justificativas_v2.jsonl")
    parser.add_argument("--categorizacao", type=Path, default=None, help="default: data/analysis/consenso_categorizacao_v2.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="default: data/analysis/consenso_verificacao_manual.jsonl")
    parser.add_argument("--n", type=int, default=N_AMOSTRA)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--manter", type=Path, default=None, help="JSON (formato da saída) com ferramentas já verificadas a preservar.")
    args = parser.parse_args()

    amostra_path = args.amostra or (DATA_DIR / "analysis" / "consenso_amostra.jsonl")
    justificativas_path = args.justificativas or (DATA_DIR / "analysis" / "consenso_justificativas_v2.jsonl")
    categorizacao_path = args.categorizacao or (DATA_DIR / "analysis" / "consenso_categorizacao_v2.jsonl")
    output_path = args.output or (DATA_DIR / "analysis" / "consenso_verificacao_manual.json")

    for p in (amostra_path, justificativas_path, categorizacao_path):
        if not p.exists():
            logger.error("%s não encontrado.", p)
            raise SystemExit(1)

    amostra: dict[str, dict] = {}
    with open(amostra_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                amostra[d["tool_uid"]] = d

    justificativas: dict[str, str] = {}
    with open(justificativas_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                justificativas[d["tool_uid"]] = d["justification"]

    categorizacao: dict[str, list[str]] = {}
    with open(categorizacao_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                categorizacao[d["tool_uid"]] = d["categories"]

    tools_por_categoria: dict[str, list[str]] = defaultdict(list)
    for tool_uid, categorias in categorizacao.items():
        if tool_uid not in amostra or tool_uid not in justificativas:
            continue
        for cat in categorias:
            tools_por_categoria[cat].append(tool_uid)

    logger.info("%s estratos (categorias + none) carregados de %s tools categorizadas", len(tools_por_categoria), len(categorizacao))

    tamanhos = {cat: len(tools) for cat, tools in tools_por_categoria.items()}
    alocacao = _alocar_por_estrato(tamanhos, args.n)
    for cat in sorted(alocacao, key=lambda c: -tamanhos[c]):
        logger.info("  %s: %s/%s vagas (população do estrato: %s)", cat, alocacao[cat], args.n, tamanhos[cat])

    rng = random.Random(args.seed)
    repos_usados: Counter[str] = Counter()
    componentes_globais: Counter[str] = Counter()
    componentes_por_estrato: dict[str, Counter[str]] = defaultdict(Counter)
    mantidas_por_estrato: Counter[str] = Counter()
    tools_selecionadas: set[str] = set()
    resultado: list[dict] = []

    if args.manter:
        anteriores = json.loads(args.manter.read_text(encoding="utf-8"))
        for item in anteriores:
            tool_uid, cat = item["tool_uid"], item["categoria"]
            if tool_uid not in amostra or cat not in categorizacao.get(tool_uid, []):
                continue
            tools_selecionadas.add(tool_uid)
            repos_usados[_repo_from_tool_uid(tool_uid)] += 1
            comps = _componentes_divergentes(amostra[tool_uid]["avaliacoes"])
            componentes_por_estrato[cat].update(comps)
            componentes_globais.update(comps)
            mantidas_por_estrato[cat] += 1
            resultado.append(item)
        logger.info("%s de %s ferramentas mantidas de %s", len(resultado), len(anteriores), args.manter)

    # Estratos menores primeiro: têm menos candidatos, então escolhem antes que um estrato
    # maior "roube" (via categoria múltipla) um dos poucos candidatos de um estrato pequeno.
    for cat in sorted(alocacao, key=lambda c: tamanhos[c]):
        vagas = max(0, alocacao[cat] - mantidas_por_estrato[cat])
        candidatos = [t for t in tools_por_categoria[cat] if t not in tools_selecionadas]
        componentes_estrato = componentes_por_estrato[cat]

        for _ in range(vagas):
            if not candidatos:
                break
            rng.shuffle(candidatos)  # desempate aleatório reprodutível (seed fixa)
            melhor, melhor_score = None, None
            for cand in candidatos:
                repo = _repo_from_tool_uid(cand)
                comps = _componentes_divergentes(amostra[cand]["avaliacoes"])
                score = 0.0
                score += 3.0 if repos_usados[repo] == 0 else -float(repos_usados[repo])
                score += len(comps - set(componentes_estrato))
                score += 0.5 * len(comps - set(componentes_globais))
                if melhor_score is None or score > melhor_score:
                    melhor, melhor_score = cand, score

            candidatos.remove(melhor)
            tools_selecionadas.add(melhor)
            repo = _repo_from_tool_uid(melhor)
            repos_usados[repo] += 1
            comps = _componentes_divergentes(amostra[melhor]["avaliacoes"])
            componentes_estrato.update(comps)
            componentes_globais.update(comps)

            avaliacoes_por_provider: dict[str, list[dict]] = defaultdict(list)
            for av in amostra[melhor]["avaliacoes"]:
                avaliacoes_por_provider[av["provider"]].append({"scenario": av["scenario"], "scores": av["scores"]})

            resultado.append(
                {
                    "tool_uid": melhor,
                    "tool_name": amostra[melhor]["tool_name"],
                    "avaliacoes": dict(avaliacoes_por_provider),
                    "justicativa": justificativas[melhor],
                    "categoria": cat,
                }
            )

    logger.info("%s ferramentas selecionadas (meta: %s)", len(resultado), args.n)
    logger.info("Repositórios distintos na amostra: %s", len(repos_usados))
    logger.info("Repositórios reaproveitados (>1 ferramenta): %s", sum(1 for v in repos_usados.values() if v > 1))
    logger.info("Distribuição de componentes divergentes na amostra: %s", dict(componentes_globais.most_common()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Amostra de verificação manual salva em %s", output_path)


if __name__ == "__main__":
    main()
