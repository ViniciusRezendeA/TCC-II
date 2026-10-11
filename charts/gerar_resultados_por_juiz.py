"""
Resultados (RQ2/RQ3) a partir da base ATUAL de avaliações, separados por juiz.

Diferente de gerar_graficos.py (que lia o snapshot charts/file.json e tirava
a MÉDIA entre modelos), este script lê diretamente data/evaluations/*.jsonl e
reporta cada juiz separadamente: os juízes reagem ao código em sentidos
opostos, e a média entre eles cancela parte do efeito.

Para cada juiz: pareia description_only x with_source pela mesma chave de
tool, deduplicação e filtro de versão de prompt de
scripts/analysis_evaluation_report.py (tool_key_for, dedupe_records,
registros_versao_ativa), e calcula, por componente: médias, % diminuiu/
empatou/aumentou, Wilcoxon (zero_method="wilcox") com correção de
Benjamini-Hochberg entre os seis componentes, e tamanho de efeito r
(correlação rank-biserial pareada, Kerby 2014).

Saídas em charts/outputs_por_juiz/ (CSVs + PNGs). Rodar da raiz do projeto:
    uv run python charts/gerar_resultados_por_juiz.py
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import false_discovery_control, rankdata, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analysis_evaluation_report import (  # noqa: E402
    load_evaluations,
    motivos_por_divergencia,
    registros_versao_ativa,
    tool_key_for,
)
from scripts.dedupe_evaluations import dedupe_records  # noqa: E402

EVAL_DIR = "data/evaluations"
OUTPUT_DIR = "charts/outputs_por_juiz"

JUDGES = {
    "deepseek-flash": "DeepSeek-V4.1-Flash",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "qwen3-14b-ollama": "Qwen3-14B",
}

ATTRIBUTES = [
    "purpose",
    "parameter_explanation",
    "length_completeness",
    "guidelines",
    "limitations",
    "examples",
]

ATTR_LABELS = {
    "purpose": "Purpose",
    "guidelines": "Guidelines",
    "limitations": "Limitations",
    "parameter_explanation": "Parameter Explanation",
    "length_completeness": "Length & Completeness",
    "examples": "Examples",
}

COR_DIMINUIU = "#d62728"
COR_EMPATOU = "#bbbbbb"
COR_AUMENTOU = "#1f77b4"

VIRGULA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "font.size": 12,
})


# ---------------------------------------------------------------------------
# 1. CARREGAR DADOS (nível juiz x tool x atributo, apenas pares completos)
# ---------------------------------------------------------------------------
def carregar_registros():
    records = dedupe_records(load_evaluations(Path(EVAL_DIR)))
    return [r for r in registros_versao_ativa(records) if r["judge"]["id"] in JUDGES]


def carregar_pares(records):
    scores = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        scores[(r["judge"]["id"], tool_key_for(r), r["scenario"])] = {
            a: r["scores"][a]["score"] for a in ATTRIBUTES
        }

    rows = []
    for (judge_id, tool, scenario), sem in scores.items():
        if scenario != "description_only":
            continue
        com = scores.get((judge_id, tool, "with_source"))
        if com is None:
            continue
        for a in ATTRIBUTES:
            rows.append({
                "judge": judge_id,
                "tool_uid": tool,
                "attribute": a,
                "score_sem_codigo": sem[a],
                "score_com_codigo": com[a],
                "diff": com[a] - sem[a],
            })
    df = pd.DataFrame(rows)
    df["judge"] = pd.Categorical(df["judge"], categories=list(JUDGES), ordered=True)
    return df.sort_values(["judge", "tool_uid"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. ESTATÍSTICAS POR JUIZ x COMPONENTE
# ---------------------------------------------------------------------------
def rank_biserial(diff):
    """Correlação rank-biserial pareada (Kerby, 2014), ignorando empates."""
    d = diff[diff != 0]
    if len(d) == 0:
        return np.nan
    r = rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def estatisticas(df):
    resultados = []
    for judge_id, g_judge in df.groupby("judge", sort=False, observed=True):
        linhas = []
        for a in ATTRIBUTES:
            g = g_judge[g_judge["attribute"] == a]
            diff = g["diff"].to_numpy()
            p = wilcoxon(g["score_com_codigo"], g["score_sem_codigo"],
                         zero_method="wilcox").pvalue
            linhas.append({
                "judge": judge_id,
                "attribute": a,
                "n_pareado": len(g),
                "media_sem_codigo": g["score_sem_codigo"].mean(),
                "media_com_codigo": g["score_com_codigo"].mean(),
                "delta": diff.mean(),
                "pct_diminuiu": (diff < 0).mean() * 100,
                "pct_empatou": (diff == 0).mean() * 100,
                "pct_aumentou": (diff > 0).mean() * 100,
                "n_diferentes": int((diff != 0).sum()),
                "r_rank_biserial": rank_biserial(diff),
                "p_valor": p,
            })
        p_bh = false_discovery_control([l["p_valor"] for l in linhas])
        for l, pb in zip(linhas, p_bh):
            l["p_valor_bh"] = pb
        resultados.extend(linhas)
    return pd.DataFrame(resultados)


def subconjunto_comum(df):
    """Restringe às tools pareadas pelos três juízes."""
    comuns = set.intersection(*[
        set(g["tool_uid"]) for _, g in df.groupby("judge", observed=True)
    ])
    return df[df["tool_uid"].isin(comuns)]


def robustez_subconjunto_comum(df):
    """Mesmo delta médio, restrito às tools pareadas pelos três juízes."""
    sub = subconjunto_comum(df)
    comuns = set(sub["tool_uid"])
    tabela = sub.groupby(["judge", "attribute"], observed=True)["diff"].mean().unstack()
    return len(comuns), tabela


def correlacao_entre_juizes(df):
    """Spearman entre cada par de juízes, por componente: nota sem código (S), nota com
    código (C) e mudança da nota (Δ = C - S). Chamado com subconjunto_comum(df), para que
    os três pares sejam comparados sobre as mesmas tools."""
    colunas = {"S": "score_sem_codigo", "C": "score_com_codigo", "Δ": "diff"}
    largos = {k: df.pivot_table(index=["tool_uid", "attribute"], columns="judge",
                                values=v, observed=True) for k, v in colunas.items()}
    linhas = []
    for a, b in combinations(JUDGES, 2):
        for medida, largo in largos.items():
            par = largo[[a, b]].dropna()
            for attr in ATTRIBUTES + ["geral"]:
                s = par if attr == "geral" else par.xs(attr, level="attribute")
                linhas.append({
                    "juiz_a": a, "juiz_b": b, "medida": medida, "attribute": attr,
                    "n_tools": len(par) // len(ATTRIBUTES),
                    "spearman": spearmanr(s[a], s[b]).statistic,
                })
    return pd.DataFrame(linhas)


def gerar_tabela_latex_correlacao(df_corr, output_path):
    pares = list(combinations(JUDGES, 2))
    nomes_curtos = {"deepseek-flash": "DeepSeek", "gemini-3.5-flash-lite": "Gemini",
                    "qwen3-14b-ollama": "Qwen3"}
    fmt = lambda v: f"{v:.2f}".replace(".", ",").replace("-", "$-$")

    cabecalho_pares = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{\\textbf{{{nomes_curtos[a]} $\\times$ {nomes_curtos[b]}}}}}"
        for a, b in pares)
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(pares)))
    linhas = []
    for attr in ATTRIBUTES + ["geral"]:
        valores = []
        for a, b in pares:
            for medida in ["S", "C", "Δ"]:
                v = df_corr.query("juiz_a == @a and juiz_b == @b and medida == @medida "
                                  "and attribute == @attr")["spearman"].iloc[0]
                valores.append(fmt(v))
        rotulo = "\\textbf{Geral}" if attr == "geral" else ATTR_LABELS[attr].replace("&", "\\&")
        if attr == "geral":
            linhas.append("\\midrule")
        linhas.append(f"{rotulo} & " + " & ".join(valores) + " \\\\")

    n_tools = f"{df_corr['n_tools'].iloc[0]:,}".replace(",", ".")
    corpo = "\n".join(linhas)
    tabela = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Correlação de Spearman ($\\rho$) entre os juízes, por componente da rubrica: "
        "nota no cenário sem código (S), nota no cenário com código (C) e mudança da nota "
        f"($\\Delta = C - S$), sobre as {n_tools} ferramentas avaliadas pelos três juízes.}}\n"
        "\\label{tab:correlacao-juizes}\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{lrrrrrrrrr}\n"
        "\\toprule\n"
        f" & {cabecalho_pares} \\\\\n"
        f"{cmid}\n"
        "\\textbf{Componente}" + " & S & C & $\\Delta$" * len(pares) + " \\\\\n"
        "\\midrule\n"
        f"{corpo}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tabela)
    print(f"[OK] {output_path}")


def gerar_tabela_latex_correlacao_geral(df_corr, output_path):
    """Versão resumida de gerar_tabela_latex_correlacao(): só a linha Geral, um par por linha."""
    nomes_curtos = {"deepseek-flash": "DeepSeek", "gemini-3.5-flash-lite": "Gemini",
                    "qwen3-14b-ollama": "Qwen3"}
    fmt = lambda v: f"{v:.2f}".replace(".", ",").replace("-", "$-$")
    geral = df_corr[df_corr["attribute"] == "geral"]

    linhas = []
    for a, b in combinations(JUDGES, 2):
        par = geral[(geral["juiz_a"] == a) & (geral["juiz_b"] == b)].set_index("medida")["spearman"]
        linhas.append(f"{nomes_curtos[a]} $\\times$ {nomes_curtos[b]} & "
                      f"{fmt(par['S'])} & {fmt(par['C'])} & {fmt(par['Δ'])} \\\\")

    n_tools = f"{df_corr['n_tools'].iloc[0]:,}".replace(",", ".")
    corpo = "\n".join(linhas)
    tabela = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Correlação de Spearman ($\\rho$) geral entre os juízes, considerando os seis "
        "componentes da rubrica em conjunto: nota no cenário sem código (S), nota no cenário com "
        f"código (C) e mudança da nota ($\\Delta = C - S$), sobre as {n_tools} ferramentas "
        "avaliadas pelos três juízes.}\n"
        "\\label{tab:correlacao-juizes-geral}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "\\textbf{Par de juízes} & \\textbf{S} & \\textbf{C} & \\textbf{$\\Delta$} \\\\\n"
        "\\midrule\n"
        f"{corpo}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tabela)
    print(f"[OK] {output_path}")


# ---------------------------------------------------------------------------
# 3. TRANSIÇÃO DE QUARTIS (faixas definidas pela distribuição sem código)
# ---------------------------------------------------------------------------
def _faixa(v, q1, q2, q3):
    return np.select([v <= q1, v <= q2, v <= q3], [0, 1, 2], default=3)


def transicao_quartis(df):
    linhas = []
    for (judge_id, a), g in df.groupby(["judge", "attribute"], sort=False, observed=True):
        sem = g["score_sem_codigo"].to_numpy()
        com = g["score_com_codigo"].to_numpy()
        q1, q2, q3 = np.percentile(sem, [25, 50, 75])
        fs, fc = _faixa(sem, q1, q2, q3), _faixa(com, q1, q2, q3)
        linhas.append({
            "judge": judge_id,
            "attribute": a,
            "pct_manteve": (fc == fs).mean() * 100,
            "pct_subiu": (fc > fs).mean() * 100,
            "pct_desceu": (fc < fs).mean() * 100,
        })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 4. MOTIVOS (busca por palavras-chave), RESUMO POR JUIZ
# ---------------------------------------------------------------------------
def resumo_motivos(records):
    m = motivos_por_divergencia(records)
    linhas = []
    for (judge_id, motivo), g in m.groupby(["juiz", "motivo"]):
        total_div = len(m.loc[m["juiz"] == judge_id, ["componente", "tool_uid"]].drop_duplicates())
        linhas.append({
            "judge": judge_id,
            "motivo": motivo,
            "ocorrencias": len(g),
            "total_divergencias": total_div,
            "pct_das_divergencias": len(g) / total_div * 100,
            "subiu": int(g["subiu"].sum()),
            "desceu": int((~g["subiu"]).sum()),
        })
    return pd.DataFrame(linhas).sort_values(["judge", "ocorrencias"], ascending=[True, False])


# ---------------------------------------------------------------------------
# 5. FIGURAS
# ---------------------------------------------------------------------------
def grafico_diminuiu_empatou_aumentou(df_stats, output_path):
    ordem = ATTRIBUTES[::-1]
    fig, axes = plt.subplots(1, len(JUDGES), figsize=(16, 5.5), sharey=True)
    for ax, (judge_id, nome) in zip(axes, JUDGES.items()):
        s = df_stats[df_stats["judge"] == judge_id].set_index("attribute").loc[ordem]
        y = np.arange(len(ordem))
        esquerda = np.zeros(len(ordem))
        for col, cor, rotulo in [("pct_diminuiu", COR_DIMINUIU, "Diminuiu"),
                                 ("pct_aumentou", COR_AUMENTOU, "Aumentou"),
                                 ("pct_empatou", COR_EMPATOU, "Empatou")]:
            valores = s[col].to_numpy()
            barras = ax.barh(y, valores, left=esquerda, color=cor, label=rotulo)
            for rect, v in zip(barras, valores):
                if v < 4:
                    continue
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + rect.get_height() / 2,
                        f"{v:.0f}%", va="center", ha="center", fontsize=9,
                        color="black" if col == "pct_empatou" else "white")
            esquerda += valores
        n = int(s["n_pareado"].iloc[0])
        ax.set_title(f"{nome}\n(n = {n:,} tools)".replace(",", "."), fontsize=12)
        ax.set_yticks(y)
        ax.set_yticklabels([ATTR_LABELS[a] for a in ordem])
        ax.set_xlim(0, 100)
        ax.set_xlabel("% de tools")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def grafico_delta_por_juiz(df_stats, output_path):
    ordem = ATTRIBUTES[::-1]
    cores = {"deepseek-flash": "#4c72b0", "gemini-3.5-flash-lite": "#dd8452",
             "qwen3-14b-ollama": "#55a868"}
    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(ordem))
    altura = 0.26
    for i, (judge_id, nome) in enumerate(JUDGES.items()):
        s = df_stats[df_stats["judge"] == judge_id].set_index("attribute").loc[ordem]
        pos = y + (1 - i) * altura
        ax.barh(pos, s["delta"], height=altura, color=cores[judge_id], label=nome)
        for yi, v in zip(pos, s["delta"]):
            ax.text(v + (0.01 if v >= 0 else -0.01), yi, f"{v:+.2f}".replace(".", ","),
                    va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([ATTR_LABELS[a] for a in ordem])
    ax.set_xlabel("Diferença média da nota (com código − sem código)")
    ax.set_xlim(-0.25, 0.72)
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = carregar_registros()
    df = carregar_pares(records)

    df_stats = estatisticas(df)
    df_stats.to_csv(os.path.join(OUTPUT_DIR, "comparacao_por_juiz.csv"), index=False)

    n_comuns, delta_comum = robustez_subconjunto_comum(df)
    delta_comum.to_csv(os.path.join(OUTPUT_DIR, "delta_subconjunto_comum.csv"))

    df_corr = correlacao_entre_juizes(subconjunto_comum(df))
    df_corr.to_csv(os.path.join(OUTPUT_DIR, "correlacao_entre_juizes.csv"), index=False)
    gerar_tabela_latex_correlacao(df_corr, os.path.join(OUTPUT_DIR, "tabela_correlacao_juizes.tex"))
    gerar_tabela_latex_correlacao_geral(
        df_corr, os.path.join(OUTPUT_DIR, "tabela_correlacao_juizes_geral.tex"))

    df_quartis = transicao_quartis(df)
    df_quartis.to_csv(os.path.join(OUTPUT_DIR, "transicao_quartis_por_juiz.csv"), index=False)

    df_motivos = resumo_motivos(records)
    df_motivos.to_csv(os.path.join(OUTPUT_DIR, "motivos_por_juiz.csv"), index=False)

    grafico_diminuiu_empatou_aumentou(
        df_stats, os.path.join(OUTPUT_DIR, "diminuiu_empatou_aumentou_por_juiz.png"))
    grafico_delta_por_juiz(df_stats, os.path.join(OUTPUT_DIR, "delta_medio_por_juiz.png"))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df_stats.round(3).to_string())
    print(f"\nSubconjunto comum aos três juízes: {n_comuns} tools")
    print(delta_comum.round(3).to_string())
    print(df_quartis.round(1).to_string())
    print(df_motivos.round(1).to_string())


if __name__ == "__main__":
    main()
