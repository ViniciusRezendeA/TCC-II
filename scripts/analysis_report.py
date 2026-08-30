from __future__ import annotations

"""Gera gráficos e planilhas comparando os resultados das Etapas 1 e 2
(mineração e extração) -- repositórios e tools por linguagem, médias por
servidor, distribuição por padrão de SDK, taxas de resolução do call graph,
concentração de tools por repositório, etc.

Deliberadamente separado do pipeline de extração: lê só os artefatos de
saída já produzidos (`data/selected_repos.jsonl`, `data/dataset.jsonl`) via
`json`/`pandas`, sem importar nada de `mcp_pipeline.extraction` (tree-sitter,
detectores de padrão, etc.). Pode ser rodado a qualquer momento depois de
`assemble_dataset.py`, quantas vezes for preciso, sem custo de API nem
reprocessamento dos repositórios.

Uso:
    uv run python -m scripts.analysis_report [--dataset PATH] [--selected-repos PATH] [--output-dir PATH]

Saída em `data/analysis/` (por padrão):
    charts/*.png              -- um gráfico por métrica, pronto para incluir no TCC
    tables/resumo_etapas_1_2.xlsx  -- uma planilha, uma aba por métrica
    tables/*.csv               -- a mesma métrica, uma por arquivo (uso em outras ferramentas)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem display -- só salva PNG, roda igual em CI/terminal remoto
import matplotlib.pyplot as plt
import pandas as pd

from mcp_pipeline.collection.dedupe_rank import dedupe as dedupe_candidates
from mcp_pipeline.collection.dedupe_rank import read_jsonl as read_candidates_jsonl
from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("analysis_report")

CHART_STYLE = {"figsize": (9, 5.5), "dpi": 160}


# --- Carregamento -----------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --- Métricas: cada função recebe os registros brutos e devolve um DataFrame
# já pronto para virar tabela/gráfico. Mantidas separadas da geração de
# gráfico/planilha para que qualquer uma possa ser reaproveitada isolada
# (ex: em um notebook de análise) sem puxar matplotlib/openpyxl junto.


def repos_por_linguagem(selected_repos: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(selected_repos)
    counts = df["primary_language"].value_counts().rename_axis("linguagem").reset_index(name="repositorios")
    counts["percentual"] = (counts["repositorios"] / len(df) * 100).round(1)
    return counts.sort_values("repositorios", ascending=False).reset_index(drop=True)


def funil_coleta(raw_candidates: list, pool: list[dict], selected: list[dict]) -> pd.DataFrame:
    """Funil de atrito da Etapa 1: quanto sobra a cada filtro aplicado, da busca bruta
    (all_candidates.jsonl -- pode ter o mesmo repositório mais de uma vez, casado por
    sub-queries/fontes de sinal diferentes) até o corte final por estrelas (top_n).
    Reusa dedupe() do próprio dedupe_rank.py em vez de recontar por conta própria, para
    esta tabela nunca divergir da lógica real de deduplicação da Etapa 1.
    """
    n_brutos = len(raw_candidates)
    n_unicos = len(dedupe_candidates(raw_candidates))
    n_pool = len(pool)
    n_selecionados = len(selected)
    df = pd.DataFrame(
        [
            {"etapa": "1. Candidatos brutos (com duplicata entre sub-queries)", "repositorios": n_brutos},
            {"etapa": "2. Únicos (deduplicados por id)", "repositorios": n_unicos},
            {"etapa": "3. Aprovados no filtro (não-fork, estrelas >= mínimo, linguagem-alvo)", "repositorios": n_pool},
            {"etapa": "4. Selecionados (top_N por estrelas)", "repositorios": n_selecionados},
        ]
    )
    df["percentual_do_bruto"] = (df["repositorios"] / n_brutos * 100).round(1) if n_brutos else 0.0
    return df


def candidatos_por_fonte(raw_candidates: list) -> pd.DataFrame:
    """Quantos candidatos brutos cada fonte de sinal (topic/text/manifest) contribuiu,
    pelo prefixo de matched_signals -- mesma lógica de
    run_step1.py::_log_candidates_per_source, exportada aqui como tabela em vez de só
    log. Um candidato casado por sinais de mais de uma fonte antes da deduplicação
    final conta uma vez em cada fonte que o encontrou (soma pode passar de 100%).
    """
    per_source: dict[str, int] = {}
    for candidate in raw_candidates:
        for signal in candidate.matched_signals:
            source = signal.split(":", 1)[0]
            per_source[source] = per_source.get(source, 0) + 1
    total = sum(per_source.values())
    rows = [
        {"fonte": source, "candidatos": n, "percentual": round(n / total * 100, 1) if total else 0.0}
        for source, n in sorted(per_source.items(), key=lambda kv: -kv[1])
    ]
    return pd.DataFrame(rows)


def tools_por_linguagem(dataset: list[dict]) -> pd.DataFrame:
    languages = [r["repo"]["primary_language"] for r in dataset]
    counts = pd.Series(languages).value_counts().rename_axis("linguagem").reset_index(name="tools")
    counts["percentual"] = (counts["tools"] / len(dataset) * 100).round(1)
    return counts.sort_values("tools", ascending=False).reset_index(drop=True)


def media_tools_por_server(selected_repos: list[dict], dataset: list[dict]) -> pd.DataFrame:
    """Duas médias por linguagem (e uma linha 'Total'), de propósito
    diferente: 'todos os repos selecionados' mostra o rendimento real da
    Etapa 2 sobre o pool inteiro da Etapa 1 (inclui repos com 0 tools --
    ou seja, mostra também o efeito de filtro da Etapa 2); 'só repos com
    >=1 tool' mostra o quão rica é a superfície de um servidor MCP real já
    confirmado, sem o 0 dos candidatos que não eram servidores de fato.
    """
    repos_by_lang = pd.Series([r["primary_language"] for r in selected_repos]).value_counts()
    tools_by_lang = pd.Series([r["repo"]["primary_language"] for r in dataset]).value_counts()
    repos_with_tools_by_lang = pd.Series(
        {r["repo"]["name_with_owner"]: r["repo"]["primary_language"] for r in dataset}
    ).value_counts()

    rows = []
    for lang in sorted(set(repos_by_lang.index) | set(tools_by_lang.index)):
        n_repos = int(repos_by_lang.get(lang, 0))
        n_tools = int(tools_by_lang.get(lang, 0))
        n_repos_with_tools = int(repos_with_tools_by_lang.get(lang, 0))
        rows.append(
            {
                "linguagem": lang,
                "repositorios_selecionados": n_repos,
                "repositorios_com_tools": n_repos_with_tools,
                "tools_total": n_tools,
                "media_tools_por_repo_selecionado": round(n_tools / n_repos, 2) if n_repos else 0.0,
                "media_tools_por_repo_com_tools": round(n_tools / n_repos_with_tools, 2) if n_repos_with_tools else 0.0,
            }
        )
    total_repos = len(selected_repos)
    total_tools = len(dataset)
    total_repos_with_tools = len({r["repo"]["name_with_owner"] for r in dataset})
    rows.append(
        {
            "linguagem": "Total",
            "repositorios_selecionados": total_repos,
            "repositorios_com_tools": total_repos_with_tools,
            "tools_total": total_tools,
            "media_tools_por_repo_selecionado": round(total_tools / total_repos, 2) if total_repos else 0.0,
            "media_tools_por_repo_com_tools": round(total_tools / total_repos_with_tools, 2)
            if total_repos_with_tools
            else 0.0,
        }
    )
    return pd.DataFrame(rows)


def distribuicao_sdk_pattern(dataset: list[dict]) -> pd.DataFrame:
    patterns = [r["tool"]["sdk_pattern"] for r in dataset]
    counts = pd.Series(patterns).value_counts().rename_axis("sdk_pattern").reset_index(name="tools")
    counts["percentual"] = (counts["tools"] / len(dataset) * 100).round(1)
    return counts.sort_values("tools", ascending=False).reset_index(drop=True)


def taxa_cobertura_por_linguagem(selected_repos: list[dict], dataset: list[dict]) -> pd.DataFrame:
    """Quantos repos selecionados na Etapa 1, por linguagem, de fato
    confirmaram ter >=1 tool detectável na Etapa 2 -- a estatística que
    sustenta a leitura de "Etapa 2 como filtro de confirmação sobre o pool
    de candidatos da Etapa 1", não só extração.
    """
    repos_by_lang = pd.Series([r["primary_language"] for r in selected_repos]).value_counts()
    repos_with_tools_by_lang = pd.Series(
        {r["repo"]["name_with_owner"]: r["repo"]["primary_language"] for r in dataset}
    ).value_counts()

    rows = []
    for lang in sorted(repos_by_lang.index):
        total = int(repos_by_lang[lang])
        with_tools = int(repos_with_tools_by_lang.get(lang, 0))
        rows.append(
            {
                "linguagem": lang,
                "repositorios_selecionados": total,
                "repositorios_com_tools": with_tools,
                "repositorios_sem_tools": total - with_tools,
                "taxa_cobertura_percentual": round(with_tools / total * 100, 1) if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("taxa_cobertura_percentual", ascending=False).reset_index(drop=True)


def _iter_call_sites(node: dict):
    """Percorre só os nós CHAMADOS (nível >= 2), não a raiz (nível 1 = a
    própria tool, sempre resolved=True/external=False por construção --
    incluí-la infla artificialmente a taxa de resolução)."""
    for child in node.get("calls", []):
        yield child
        yield from _iter_call_sites(child)


def call_graph_resolucao(dataset: list[dict]) -> pd.DataFrame:
    """`ambiguous` não é uma categoria própria no schema -- é um flag
    ortogonal que também pode estar `True` num nó `resolved`. Para a tabela
    virar uma partição de fato mutuamente exclusiva (percentuais somando
    100%, sem parecer um erro de conta num documento), as 3 categorias
    exportadas aqui são recombinadas explicitamente: resolvido sem ambiguidade,
    resolvido por desempate ambíguo, e externo/não resolvido.
    """
    total = resolved_unambiguous = resolved_ambiguous = external = 0
    by_level: dict[int, int] = {}
    for r in dataset:
        for node in _iter_call_sites(r["call_graph"]):
            total += 1
            by_level[node["level"]] = by_level.get(node["level"], 0) + 1
            if node["resolved"] and node["ambiguous"]:
                resolved_ambiguous += 1
            elif node["resolved"]:
                resolved_unambiguous += 1
            else:
                external += 1

    rows = [
        {"categoria": "Resolvido, sem ambiguidade", "nos": resolved_unambiguous,
         "percentual": round(resolved_unambiguous / total * 100, 1) if total else 0.0},
        {"categoria": "Resolvido, com desempate ambíguo", "nos": resolved_ambiguous,
         "percentual": round(resolved_ambiguous / total * 100, 1) if total else 0.0},
        {"categoria": "Externo/dinâmico (não resolvido)", "nos": external,
         "percentual": round(external / total * 100, 1) if total else 0.0},
    ]
    df = pd.DataFrame(rows)
    df.attrs["total_nos_chamada"] = total
    df.attrs["nivel_2"] = by_level.get(2, 0)
    df.attrs["nivel_3"] = by_level.get(3, 0)
    return df


def description_literal_rate(dataset: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(
        [{"linguagem": r["repo"]["primary_language"], "literal": r["tool"]["description_is_literal"]} for r in dataset]
    )
    grouped = df.groupby("linguagem")["literal"].agg(["sum", "count"]).reset_index()
    grouped.columns = ["linguagem", "descricoes_literais", "tools_total"]
    grouped["taxa_literal_percentual"] = (grouped["descricoes_literais"] / grouped["tools_total"] * 100).round(1)
    total_row = pd.DataFrame(
        [{
            "linguagem": "Total",
            "descricoes_literais": int(df["literal"].sum()),
            "tools_total": len(df),
            "taxa_literal_percentual": round(df["literal"].sum() / len(df) * 100, 1) if len(df) else 0.0,
        }]
    )
    return pd.concat([grouped.sort_values("taxa_literal_percentual", ascending=False), total_row], ignore_index=True)


def top_repos_por_tools(dataset: list[dict], n: int = 15) -> pd.DataFrame:
    counts = pd.Series([r["repo"]["name_with_owner"] for r in dataset]).value_counts().head(n)
    lang_by_repo = {r["repo"]["name_with_owner"]: r["repo"]["primary_language"] for r in dataset}
    stars_by_repo = {r["repo"]["name_with_owner"]: r["repo"]["stargazer_count"] for r in dataset}
    rows = [
        {"repositorio": repo, "tools": int(count), "linguagem": lang_by_repo[repo], "estrelas": stars_by_repo[repo]}
        for repo, count in counts.items()
    ]
    df = pd.DataFrame(rows)
    total_tools = len(dataset)
    df["percentual_do_total"] = (df["tools"] / total_tools * 100).round(1)
    return df


def distribuicao_tools_por_repo(dataset: list[dict]) -> pd.Series:
    """Uma linha por repositório com >=1 tool -- usada só para o histograma
    de concentração, não exportada como tabela própria (top_repos_por_tools
    já cobre o mesmo dado de forma mais legível em planilha)."""
    return pd.Series([r["repo"]["name_with_owner"] for r in dataset]).value_counts()


def distribuicao_estrelas(selected_repos: list[dict]) -> pd.DataFrame:
    stars = pd.Series([r["stargazer_count"] for r in selected_repos])
    return pd.DataFrame(
        {
            "estatistica": ["mínimo", "p25", "mediana", "p75", "máximo", "média"],
            "estrelas": [
                int(stars.min()), int(stars.quantile(0.25)), int(stars.median()),
                int(stars.quantile(0.75)), int(stars.max()), round(stars.mean(), 1),
            ],
        }
    )


def _loc_series(dataset: list[dict]) -> pd.Series:
    """Uma linha por tool -- usada só para o histograma; distribuicao_loc()
    já cobre o resumo estatístico em planilha (mesmo padrão de
    distribuicao_tools_por_repo() vs. top_repos_por_tools())."""
    return pd.Series([r["tool"]["loc"] for r in dataset])


def distribuicao_loc(dataset: list[dict]) -> pd.DataFrame:
    """Estatísticas de LOC da função que implementa cada tool -- computado em
    tool_detector.py a partir do FunctionDef resolvido (start_def), não de
    tool["source_location"] (que, nos padrões .tool()/.registerTool() de
    JS/TS, é o call site de registro, não o corpo do handler). Nota: em
    Java/C#, o nó do método inclui a linha de anotação/atributo
    (@Tool/[McpServerTool]), o que infla o LOC em ~1 linha por tool
    relativo a Python -- diferença de gramática, não de tamanho real."""
    loc = _loc_series(dataset)
    return pd.DataFrame(
        {
            "estatistica": ["mínimo", "p25", "mediana", "p75", "máximo", "média"],
            "loc": [
                int(loc.min()), int(loc.quantile(0.25)), int(loc.median()),
                int(loc.quantile(0.75)), int(loc.max()), round(loc.mean(), 1),
            ],
        }
    )


def profundidade_call_graph(dataset: list[dict]) -> pd.DataFrame:
    """Profundidade do call graph de cada tool (maior `level` alcançado).
    Limitada a {1, 2, 3} por construção -- MAX_LEVEL em call_graph_builder.py
    trava a árvore em 3 níveis; ver a docstring de build_call_graph()."""
    depths = pd.Series([r["tool"]["call_graph_depth"] for r in dataset])
    counts = depths.value_counts().rename_axis("profundidade").reset_index(name="tools")
    counts["percentual"] = (counts["tools"] / len(dataset) * 100).round(1)
    return counts.sort_values("profundidade").reset_index(drop=True)


# --- Gráficos ----------------------------------------------------------------


def _bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, path: Path, horizontal: bool = False) -> None:
    fig, ax = plt.subplots(figsize=CHART_STYLE["figsize"])
    if horizontal:
        ax.barh(df[x_col], df[y_col], color="#3b6ea5")
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        ax.bar(df[x_col], df[y_col], color="#3b6ea5")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=35, ha="right")
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def _grouped_bar_chart(df: pd.DataFrame, x_col: str, y_cols: list[str], labels: list[str], title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=CHART_STYLE["figsize"])
    x = range(len(df))
    width = 0.8 / len(y_cols)
    colors = ["#3b6ea5", "#e07b39", "#4a9e6f"]
    for i, (col, label) in enumerate(zip(y_cols, labels)):
        offset = (i - (len(y_cols) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], df[col], width=width, label=label, color=colors[i % len(colors)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(df[x_col], rotation=35, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def _histogram(
    series: pd.Series, bins, title: str, xlabel: str, ylabel: str, path: Path,
    log_x: bool = False, horizontal: bool = False,
) -> None:
    """`xlabel`/`ylabel` always name the value axis (bins) and the count axis,
    respectively, regardless of orientation -- `horizontal` only swaps which
    physical axis (x or y) each ends up on, so call sites don't need to know
    or care which orientation they'll render in."""
    fig, ax = plt.subplots(figsize=CHART_STYLE["figsize"])
    if horizontal:
        ax.hist(series, bins=bins, color="#3b6ea5", edgecolor="white", orientation="horizontal")
        if log_x:
            ax.set_yscale("log")
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    else:
        ax.hist(series, bins=bins, color="#3b6ea5", edgecolor="white")
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def _boxplot(
    series: pd.Series, title: str, value_label: str, path: Path, horizontal: bool = False, log_scale: bool = False
) -> None:
    """Real quartiles/whiskers/outliers computed by matplotlib from the raw
    series (Tukey's 1.5*IQR rule), not a hand-drawn box from pre-aggregated
    stats -- so outliers like a single wildly popular repo actually show up
    as flier points instead of stretching the whiskers to the extremes."""
    fig, ax = plt.subplots(figsize=CHART_STYLE["figsize"])
    bp = ax.boxplot(series, vert=not horizontal, patch_artist=True, widths=0.35)
    for box in bp["boxes"]:
        box.set(facecolor="#3b6ea5", alpha=0.55, edgecolor="#3b6ea5", linewidth=1.4)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set(color="#3b6ea5", linewidth=1.4)
    for line in bp["medians"]:
        line.set(color="#1f3f5c", linewidth=1.8)
    for flier in bp["fliers"]:
        flier.set(marker="o", markerfacecolor="#3b6ea5", markeredgecolor="#3b6ea5", markersize=5, alpha=0.5)
    if horizontal:
        if log_scale:
            ax.set_xscale("log")
        ax.set_yticks([])
        ax.set_xlabel(value_label)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    else:
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks([])
        ax.set_ylabel(value_label)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def generate_charts(
    tables: dict[str, pd.DataFrame], dataset: list[dict], charts_dir: Path, selected_repos: list[dict] | None = None
) -> None:
    """Cada bloco é condicional à tabela correspondente estar em `tables` -- desde que
    dataset.jsonl passou a ser opcional (ver main()), as tabelas/gráficos da Etapa 2 podem
    legitimamente estar ausentes numa rodada só de Etapa 1 (ex: antes da Etapa 2 ter sido
    (re)executada)."""
    charts_dir.mkdir(parents=True, exist_ok=True)

    if "repos_por_linguagem" in tables:
        _bar_chart(
            tables["repos_por_linguagem"], "linguagem", "repositorios",
            "Repositórios selecionados por linguagem (Etapa 1)", "Linguagem", "Repositórios",
            charts_dir / "01_repos_por_linguagem.png",
        )
    if "tools_por_linguagem" in tables:
        _bar_chart(
            tables["tools_por_linguagem"], "linguagem", "tools",
            "Tools extraídas por linguagem (Etapa 2)", "Linguagem", "Tools",
            charts_dir / "02_tools_por_linguagem.png",
        )
    if "media_tools_por_server" in tables:
        _grouped_bar_chart(
            tables["media_tools_por_server"][tables["media_tools_por_server"]["linguagem"] != "Total"],
            "linguagem",
            ["media_tools_por_repo_selecionado", "media_tools_por_repo_com_tools"],
            ["Por repo selecionado (Etapa 1)", "Por repo com >=1 tool"],
            "Média de tools por servidor, por linguagem", "Linguagem", "Média de tools",
            charts_dir / "03_media_tools_por_server.png",
        )
    if "distribuicao_sdk_pattern" in tables:
        _bar_chart(
            tables["distribuicao_sdk_pattern"], "sdk_pattern", "tools",
            "Distribuição de tools por padrão de SDK (sdk_pattern)", "Padrão", "Tools",
            charts_dir / "04_distribuicao_sdk_pattern.png", horizontal=True,
        )
    if "taxa_cobertura_por_linguagem" in tables:
        _bar_chart(
            tables["taxa_cobertura_por_linguagem"], "linguagem", "taxa_cobertura_percentual",
            "Taxa de confirmação da Etapa 2 por linguagem\n(% de repositórios selecionados com >=1 tool detectada)",
            "Linguagem", "% com >=1 tool",
            charts_dir / "05_taxa_cobertura_por_linguagem.png",
        )
    if "call_graph_resolucao" in tables:
        _bar_chart(
            tables["call_graph_resolucao"], "categoria", "percentual",
            "Resolução dos nós do call graph (níveis 2-3)", "", "% dos nós",
            charts_dir / "06_call_graph_resolucao.png", horizontal=True,
        )
    if dataset:
        _histogram(
            distribuicao_tools_por_repo(dataset), bins=30,
            title="Distribuição de tools por repositório\n(entre os repositórios com >=1 tool)",
            xlabel="Tools no repositório (escala log)", ylabel="Nº de repositórios",
            path=charts_dir / "07_distribuicao_tools_por_repo.png", log_x=True,
        )
        _histogram(
            _loc_series(dataset), bins=30,
            title="Distribuição de LOC (linhas de código) por tool\n(função implementadora, nível 1 do call graph)",
            xlabel="LOC", ylabel="Nº de tools",
            path=charts_dir / "08_distribuicao_loc.png", horizontal=True,
        )
    if "profundidade_call_graph" in tables:
        _bar_chart(
            tables["profundidade_call_graph"], "profundidade", "tools",
            "Profundidade do call graph por tool\n(nível máximo alcançado; limitado a 3 por construção — MAX_LEVEL)",
            "Profundidade (níveis)", "Tools",
            charts_dir / "09_distribuicao_profundidade_call_graph.png",
        )
    if "funil_coleta" in tables:
        _bar_chart(
            tables["funil_coleta"], "etapa", "repositorios",
            "Funil de atrito da Etapa 1\n(bruto -> único -> aprovado no filtro -> selecionado)",
            "", "Repositórios",
            charts_dir / "10_funil_coleta.png", horizontal=True,
        )
    if "candidatos_por_fonte" in tables:
        _bar_chart(
            tables["candidatos_por_fonte"], "fonte", "candidatos",
            "Candidatos brutos por fonte de sinal (topic/text/manifest)", "Fonte", "Candidatos",
            charts_dir / "11_candidatos_por_fonte.png",
        )
    if selected_repos:
        _boxplot(
            pd.Series([r["stargazer_count"] for r in selected_repos]),
            "Distribuição de estrelas dos repositórios selecionados (Etapa 1)",
            "Estrelas (escala log)",
            charts_dir / "12_distribuicao_estrelas.png", horizontal=True, log_scale=True,
        )

    logger.info("Gráficos salvos em %s", charts_dir)


# --- Planilhas ----------------------------------------------------------------


def export_tables(tables: dict[str, pd.DataFrame], tables_dir: Path, workbook_name: str = "resumo_etapas_1_2.xlsx") -> None:
    """`workbook_name` is overridable so scripts/analysis_evaluation_report.py (Etapa 3) can
    reuse this instead of duplicating the CSV/XLSX export logic, without colliding with this
    module's own resumo_etapas_1_2.xlsx."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    workbook_path = tables_dir / workbook_name
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)  # limite de 31 chars do Excel para nome de aba
    logger.info("Planilha combinada salva em %s", workbook_path)

    for name, df in tables.items():
        csv_path = tables_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("%s planilhas CSV individuais salvas em %s", len(tables), tables_dir)


# --- CLI -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera gráficos e planilhas de validação das Etapas 1 e 2 a partir de data/dataset.jsonl"
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Caminho para dataset.jsonl (default: data/dataset.jsonl).")
    parser.add_argument(
        "--selected-repos", type=Path, default=None, help="Caminho para selected_repos.jsonl (default: data/selected_repos.jsonl)."
    )
    parser.add_argument(
        "--raw-candidates", type=Path, default=None,
        help="Caminho para all_candidates.jsonl (default: data/raw/all_candidates.jsonl).",
    )
    parser.add_argument(
        "--candidate-pool", type=Path, default=None, help="Caminho para candidate_pool.jsonl (default: data/candidate_pool.jsonl)."
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretório de saída (default: data/analysis).")
    args = parser.parse_args()

    dataset_path = args.dataset or (DATA_DIR / "dataset.jsonl")
    selected_repos_path = args.selected_repos or (DATA_DIR / "selected_repos.jsonl")
    raw_candidates_path = args.raw_candidates or (DATA_DIR / "raw" / "all_candidates.jsonl")
    candidate_pool_path = args.candidate_pool or (DATA_DIR / "candidate_pool.jsonl")
    output_dir = args.output_dir or (DATA_DIR / "analysis")

    if not selected_repos_path.exists():
        logger.error("selected_repos.jsonl não encontrado em %s -- rode a Etapa 1 primeiro.", selected_repos_path)
        sys.exit(1)
    selected_repos = load_jsonl(selected_repos_path)

    # dataset.jsonl é opcional aqui: sem ele, ainda dá pra gerar as tabelas que dependem só
    # da Etapa 1 (ex: antes da Etapa 2 ter sido (re)executada) -- só as que dependem de
    # tools extraídas ficam de fora, com aviso.
    dataset: list[dict] = []
    if dataset_path.exists():
        dataset = load_jsonl(dataset_path)
    else:
        logger.warning(
            "dataset.jsonl não encontrado em %s -- pulando tabelas/gráficos da Etapa 2 "
            "(rode a Etapa 2 + assemble_dataset.py para gerá-las).",
            dataset_path,
        )

    logger.info("Carregados %s repositórios selecionados e %s tools.", len(selected_repos), len(dataset))

    tables: dict[str, pd.DataFrame] = {
        "repos_por_linguagem": repos_por_linguagem(selected_repos),
        "distribuicao_estrelas": distribuicao_estrelas(selected_repos),
    }

    if raw_candidates_path.exists() and candidate_pool_path.exists():
        raw_candidates = read_candidates_jsonl(raw_candidates_path)
        candidate_pool = load_jsonl(candidate_pool_path)
        tables["funil_coleta"] = funil_coleta(raw_candidates, candidate_pool, selected_repos)
        tables["candidatos_por_fonte"] = candidatos_por_fonte(raw_candidates)
    else:
        logger.warning(
            "all_candidates.jsonl e/ou candidate_pool.jsonl não encontrados -- pulando "
            "funil_coleta e candidatos_por_fonte (esperados em %s e %s).",
            raw_candidates_path,
            candidate_pool_path,
        )

    if dataset:
        tables.update(
            {
                "tools_por_linguagem": tools_por_linguagem(dataset),
                "media_tools_por_server": media_tools_por_server(selected_repos, dataset),
                "distribuicao_sdk_pattern": distribuicao_sdk_pattern(dataset),
                "taxa_cobertura_por_linguagem": taxa_cobertura_por_linguagem(selected_repos, dataset),
                "call_graph_resolucao": call_graph_resolucao(dataset),
                "description_literal_rate": description_literal_rate(dataset),
                "top_repos_por_tools": top_repos_por_tools(dataset),
                "distribuicao_loc": distribuicao_loc(dataset),
                "profundidade_call_graph": profundidade_call_graph(dataset),
            }
        )

    export_tables(tables, output_dir / "tables")
    generate_charts(tables, dataset, output_dir / "charts", selected_repos)

    logger.info(
        "Concluído: %s repositórios selecionados, %s tools, tabelas geradas: %s.",
        len(selected_repos),
        len(dataset),
        ", ".join(tables.keys()),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Geração do relatório de análise falhou")
        sys.exit(1)
