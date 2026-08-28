from __future__ import annotations

"""Gera gráficos e planilhas da Etapa 3 (avaliação via LLM-as-a-Judge): confiabilidade
técnica do lote (ok/refused/error), notas médias por componente/juiz/cenário, comparação
description_only vs with_source (a pergunta de pesquisa central do TCC), concordância entre
juízes, custo/latência e notas por linguagem do repositório.

Companion de scripts/analysis_report.py (Etapas 1-2): mesma convenção de saída
(charts/*.png, tables/*.csv, uma planilha resumo_etapa_3.xlsx em vez de
resumo_etapas_1_2.xlsx), reaproveitando os helpers de gráfico/planilha de lá em vez de
duplicá-los.

Lê só data/evaluations/{judge_id}.jsonl (saída de pipeline/run_step3.py) -- não faz
nenhuma chamada de API nem depende de mcp_pipeline.evaluation.judges.*. Não precisa cruzar
com data/dataset.jsonl: repo.primary_language já vem denormalizado em cada registro de
avaliação (ver pipeline/run_step3.py::_base_record).

**Ainda não rodado contra dados reais** -- a Etapa 3 (README) ainda não foi executada
contra a API real dos 3 provedores. Testado só com um fixture sintético (ver
scratchpad/local durante o desenvolvimento); rodar de verdade e conferir os números assim
que `data/evaluations/*.jsonl` existir.

As tabelas aqui são descritivas (médias, medianas, contagens, correlação). O teste de
significância pareado (Wilcoxon, tool a tool, cenário A vs B) é da Etapa 5 da metodologia
do TCC, ainda não implementada -- ver o comentário sobre isso em
pipeline/run_step3.py::tool_uid_for.

Uso:
    uv run python -m scripts.analysis_evaluation_report [--evaluations-dir PATH] [--output-dir PATH]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.logging_setup import setup_logging
from scripts.analysis_report import (
    _bar_chart,
    _grouped_bar_chart,
    export_tables,
    load_jsonl,
)

logger = setup_logging("analysis_evaluation_report")

RUBRIC_COMPONENTS = [
    "purpose",
    "guidelines",
    "limitations",
    "parameter_explanation",
    "length_completeness",
    "examples",
]


# --- Carregamento -------------------------------------------------------------


def load_evaluations(evaluations_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(evaluations_dir.glob("*.jsonl")):
        records.extend(load_jsonl(path))
    return records


def scores_long(records: list[dict]) -> pd.DataFrame:
    """Formato longo (uma linha por tool x cenário x juiz x componente), só para
    avaliações com status "ok" -- base compartilhada pela maioria das métricas abaixo, em
    vez de cada uma reimplementar sua própria navegação em scores["componente"]["score"].
    status "refused"/"error" não têm scores e ficam de fora por construção; ver
    status_por_juiz_cenario() para a contagem dessas.
    """
    rows = []
    for r in records:
        if r["status"] != "ok":
            continue
        for component in RUBRIC_COMPONENTS:
            rows.append(
                {
                    "tool_uid": r["tool_uid"],
                    "cenario": r["scenario"],
                    "juiz": r["judge"]["id"],
                    "linguagem": r["repo"].get("primary_language"),
                    "componente": component,
                    "nota": r["scores"][component]["score"],
                }
            )
    return pd.DataFrame(rows)


# --- Métricas -------------------------------------------------------------------


def status_por_juiz_cenario(records: list[dict]) -> pd.DataFrame:
    """Confiabilidade técnica do lote: quantas avaliações terminaram ok/refused/error, por
    juiz e cenário. "refused" (recusa de segurança do provedor) é um resultado esperado
    para uma fração do dataset (ex: tools de servidores de pentest), não o mesmo que
    "error" (falha técnica) -- ver JudgeRefusal em evaluation/judges/base.py.
    """
    df = pd.DataFrame([{"juiz": r["judge"]["id"], "cenario": r["scenario"], "status": r["status"]} for r in records])
    counts = df.groupby(["juiz", "cenario", "status"]).size().unstack(fill_value=0).reset_index()
    for col in ("ok", "refused", "error"):
        if col not in counts.columns:
            counts[col] = 0
    counts["total"] = counts["ok"] + counts["refused"] + counts["error"]
    counts["taxa_ok_percentual"] = (counts["ok"] / counts["total"] * 100).round(1)
    return counts[["juiz", "cenario", "ok", "refused", "error", "total", "taxa_ok_percentual"]]


def notas_por_componente(long_df: pd.DataFrame) -> pd.DataFrame:
    """Tabela central de resultado da Etapa 3: nota média (Likert 1-5) de cada um dos 6
    componentes da rubrica, por juiz e cenário."""
    grouped = long_df.groupby(["juiz", "cenario", "componente"])["nota"].agg(["mean", "median", "std", "count"]).reset_index()
    grouped.columns = ["juiz", "cenario", "componente", "media", "mediana", "desvio_padrao", "n"]
    for col in ("media", "mediana", "desvio_padrao"):
        grouped[col] = grouped[col].round(2)
    return grouped.sort_values(["componente", "juiz", "cenario"]).reset_index(drop=True)


def comparacao_cenarios(long_df: pd.DataFrame) -> pd.DataFrame:
    """A pergunta de pesquisa central do TCC: dar acesso ao SOURCE_CODE (cenário
    with_source) muda a nota, por componente e por juiz, em relação a só a descrição
    (description_only)? `n_pareado` conta só as tools avaliadas em AMBOS os cenários pelo
    mesmo juiz (join por tool_uid) -- a base correta para uma futura análise pareada
    (Wilcoxon, Etapa 5), não o total de linhas de cada cenário isoladamente.
    """
    pivot = long_df.pivot_table(index=["juiz", "componente", "tool_uid"], columns="cenario", values="nota")
    rows = []
    for (juiz, componente), group in pivot.groupby(level=["juiz", "componente"]):
        desc = group["description_only"].dropna() if "description_only" in group else pd.Series(dtype=float)
        src = group["with_source"].dropna() if "with_source" in group else pd.Series(dtype=float)
        n_pareado = int(group.dropna().shape[0])
        rows.append(
            {
                "juiz": juiz,
                "componente": componente,
                "media_description_only": round(desc.mean(), 2) if len(desc) else None,
                "media_with_source": round(src.mean(), 2) if len(src) else None,
                "delta_with_source_menos_description_only": (
                    round(src.mean() - desc.mean(), 2) if len(desc) and len(src) else None
                ),
                "n_pareado": n_pareado,
            }
        )
    return pd.DataFrame(rows).sort_values(["componente", "juiz"]).reset_index(drop=True)


def concordancia_entre_juizes(long_df: pd.DataFrame) -> pd.DataFrame:
    """Concordância par-a-par entre juízes: correlação de Pearson e diferença média
    absoluta das notas dadas ao mesmo (tool, cenário, componente) -- indica se o júri
    multi-provedor converge ou se cada modelo julga de forma sistematicamente diferente.
    """
    wide = long_df.pivot_table(index=["tool_uid", "cenario", "componente"], columns="juiz", values="nota")
    juizes = sorted(wide.columns)
    rows = []
    for i, a in enumerate(juizes):
        for b in juizes[i + 1 :]:
            pair = wide[[a, b]].dropna()
            if pair.empty:
                continue
            rows.append(
                {
                    "juiz_a": a,
                    "juiz_b": b,
                    "n_comparacoes": len(pair),
                    "correlacao_pearson": round(pair[a].corr(pair[b]), 2),
                    "diferenca_media_absoluta": round((pair[a] - pair[b]).abs().mean(), 2),
                }
            )
    return pd.DataFrame(rows)


def custo_latencia_por_juiz(records: list[dict]) -> pd.DataFrame:
    """Tokens e latência médios por juiz -- só entre avaliações "ok" (usage/latency não
    são preenchidos para refused/error). Insumo para a seção de metodologia/limitações do
    TCC (custo e tempo de execução da Etapa 3)."""
    rows = [
        {
            "juiz": r["judge"]["id"],
            "input_tokens": r["usage"]["input_tokens"],
            "output_tokens": r["usage"]["output_tokens"],
            "cache_read_input_tokens": r["usage"]["cache_read_input_tokens"],
            "latency_ms": r["latency_ms"],
        }
        for r in records
        if r["status"] == "ok"
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "juiz", "avaliacoes", "media_input_tokens", "media_output_tokens",
                "media_cache_read_tokens", "media_latencia_ms", "mediana_latencia_ms",
            ]
        )
    grouped = (
        df.groupby("juiz")
        .agg(
            avaliacoes=("latency_ms", "count"),
            media_input_tokens=("input_tokens", "mean"),
            media_output_tokens=("output_tokens", "mean"),
            media_cache_read_tokens=("cache_read_input_tokens", "mean"),
            media_latencia_ms=("latency_ms", "mean"),
            mediana_latencia_ms=("latency_ms", "median"),
        )
        .round(1)
        .reset_index()
    )
    return grouped


def notas_por_linguagem(long_df: pd.DataFrame) -> pd.DataFrame:
    """Nota média (todos os componentes juntos) por linguagem do repositório -- só
    descritivo; não controla por juiz/cenário nem tenta atribuir causalidade."""
    grouped = long_df.groupby("linguagem")["nota"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["linguagem", "nota_media", "n"]
    grouped["nota_media"] = grouped["nota_media"].round(2)
    return grouped.sort_values("nota_media", ascending=False).reset_index(drop=True)


# --- Gráficos ----------------------------------------------------------------


def generate_charts(tables: dict[str, pd.DataFrame], charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    status = tables["status_por_juiz_cenario"]
    if not status.empty:
        by_juiz = status.groupby("juiz")[["ok", "total"]].sum().reset_index()
        by_juiz["taxa_ok_percentual"] = (by_juiz["ok"] / by_juiz["total"] * 100).round(1)
        _bar_chart(
            by_juiz, "juiz", "taxa_ok_percentual",
            "Taxa de avaliações concluídas (status ok) por juiz\n(agregado entre os 2 cenários)",
            "Juiz", "% ok",
            charts_dir / "01_taxa_ok_por_juiz.png",
        )

    componente_df = tables["notas_por_componente"]
    if not componente_df.empty:
        nota_por_componente = componente_df.groupby("componente")["media"].mean().round(2).reset_index()
        nota_por_componente.columns = ["componente", "nota_media"]
        _bar_chart(
            nota_por_componente, "componente", "nota_media",
            "Nota média por componente da rubrica\n(escala Likert 1-5; média entre juízes e cenários)",
            "Componente", "Nota média",
            charts_dir / "02_nota_media_por_componente.png",
        )

    comparacao = tables["comparacao_cenarios"]
    if not comparacao.empty:
        by_componente = (
            comparacao.groupby("componente")[["media_description_only", "media_with_source"]].mean().round(2).reset_index()
        )
        _grouped_bar_chart(
            by_componente, "componente",
            ["media_description_only", "media_with_source"],
            ["Só descrição", "Com código-fonte"],
            "Nota média por componente: descrição vs. com código-fonte\n(média entre juízes)",
            "Componente", "Nota média",
            charts_dir / "03_comparacao_cenarios_por_componente.png",
        )

    concordancia = tables["concordancia_entre_juizes"]
    if not concordancia.empty:
        pares = concordancia.copy()
        pares["par"] = pares["juiz_a"] + " x " + pares["juiz_b"]
        _bar_chart(
            pares, "par", "correlacao_pearson",
            "Concordância entre pares de juízes\n(correlação de Pearson, mesmo tool/cenário/componente)",
            "Par de juízes", "Correlação de Pearson",
            charts_dir / "04_concordancia_entre_juizes.png",
        )

    notas_lang = tables["notas_por_linguagem"]
    if not notas_lang.empty:
        _bar_chart(
            notas_lang, "linguagem", "nota_media",
            "Nota média (todos os componentes) por linguagem do repositório", "Linguagem", "Nota média",
            charts_dir / "05_notas_por_linguagem.png", horizontal=True,
        )

    custo = tables["custo_latencia_por_juiz"]
    if not custo.empty:
        _bar_chart(
            custo, "juiz", "media_latencia_ms",
            "Latência média por juiz\n(ms, só avaliações concluídas)", "Juiz", "Latência média (ms)",
            charts_dir / "06_latencia_por_juiz.png",
        )

    logger.info("Gráficos salvos em %s", charts_dir)


# --- CLI -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera gráficos e planilhas de validação da Etapa 3 a partir de data/evaluations/*.jsonl")
    parser.add_argument("--evaluations-dir", type=Path, default=None, help="Diretório com {judge_id}.jsonl (default: data/evaluations).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretório de saída (default: data/analysis).")
    args = parser.parse_args()

    evaluations_dir = args.evaluations_dir or (DATA_DIR / "evaluations")
    output_dir = args.output_dir or (DATA_DIR / "analysis")

    if not evaluations_dir.exists() or not any(evaluations_dir.glob("*.jsonl")):
        logger.error(
            "Nenhum {judge_id}.jsonl encontrado em %s -- rode a Etapa 3 "
            "(pipeline/run_step3.py) primeiro.",
            evaluations_dir,
        )
        sys.exit(1)

    records = load_evaluations(evaluations_dir)
    n_ok = sum(1 for r in records if r["status"] == "ok")
    logger.info(
        "Carregadas %s avaliações (tool x cenário x juiz) de %s, %s com status ok",
        len(records), evaluations_dir, n_ok,
    )

    long_df = scores_long(records)

    tables = {
        "status_por_juiz_cenario": status_por_juiz_cenario(records),
        "notas_por_componente": notas_por_componente(long_df),
        "comparacao_cenarios": comparacao_cenarios(long_df),
        "concordancia_entre_juizes": concordancia_entre_juizes(long_df),
        "custo_latencia_por_juiz": custo_latencia_por_juiz(records),
        "notas_por_linguagem": notas_por_linguagem(long_df),
    }

    export_tables(tables, output_dir / "tables", workbook_name="resumo_etapa_3.xlsx")
    generate_charts(tables, output_dir / "charts")

    logger.info("Concluído: %s avaliações carregadas (%s ok), tabelas geradas: %s.", len(records), n_ok, ", ".join(tables.keys()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Geração do relatório de avaliação (Etapa 3) falhou")
        sys.exit(1)
